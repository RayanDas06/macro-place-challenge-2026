"""
Coordinate-descent + orientation-search polisher for macro placement.

Algorithm
---------
1.  Position CD (all macros batched together):
    • For each macro M, sample K=16 positions on a 4×4 grid at scale s.
    • Batch ALL macro candidates into chunks of BATCH=64 per GPU call.
    • Compare costs against the pre-pass baseline; accept any greedy
      improvement that satisfies the spacing constraint.
    • Scale sequence: coarse (10%) → medium (3%) → fine (1%).

2.  Orientation search (all macros batched together, 4 orientations each):
    • For each macro, generate 4 placements at current position with
      different ep_offset transforms (N / FN / S / FS).
    • Batch across macros (16 macros × 4 orients = 64 per GPU call).
    • Accept the best orientation if it beats the current cost.

3.  LNS (every lns_every passes):
    • Pick a cluster of ~8 nearby macros, rip, greedy-repack, accept if
      cost improves.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Klein-4 orientation group
# index  name   scale_x  scale_y
#   0     N       +1       +1
#   1     FN      -1       +1
#   2     S       -1       -1
#   3     FS      +1       -1
# ---------------------------------------------------------------------------
_SX = np.array([ 1., -1., -1.,  1.])
_SY = np.array([ 1.,  1., -1., -1.])
_SX_T = torch.tensor(_SX, dtype=torch.float32)
_SY_T = torch.tensor(_SY, dtype=torch.float32)

# Klein-4 multiplication: _K4[a][b] = compose(a, b)
_K4 = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
ORIENT_NAMES = ["N", "FN", "S", "FS"]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _sep(sizes: np.ndarray, extra: float = 0.0):
    sx = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0 + extra
    sy = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0 + extra
    return sx, sy


def _overlap(pos: np.ndarray, idx: int, sx: np.ndarray, sy: np.ndarray) -> bool:
    dx = np.abs(pos[:, 0] - pos[idx, 0])
    dy = np.abs(pos[:, 1] - pos[idx, 1])
    ov = (dx < sx[idx]) & (dy < sy[idx])
    ov[idx] = False
    return bool(ov.any())


# ---------------------------------------------------------------------------
# Candidate grids
# ---------------------------------------------------------------------------

def _grid(spread: float, k_side: int = 4) -> np.ndarray:
    """Return k_side² symmetric (dx,dy) offsets spanning ±spread."""
    t = np.linspace(-spread, spread, k_side)
    xs, ys = np.meshgrid(t, t, indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], axis=1)  # [K, 2]


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

def _make_batch(
    pos: np.ndarray,        # [n_hard, 2]
    soft_t: torch.Tensor,   # [n_soft, 2]
    bm: np.ndarray,         # [B] macro indices to move
    bxy: np.ndarray,        # [B, 2] new positions
    device: str,
) -> torch.Tensor:
    B = len(bm)
    hard_t = torch.tensor(pos, dtype=torch.float32, device=device)
    full = hard_t.unsqueeze(0).expand(B, -1, -1).clone()
    bi = torch.arange(B, device=device)
    mi = torch.tensor(bm, dtype=torch.long, device=device)
    full[bi, mi] = torch.tensor(bxy, dtype=torch.float32, device=device)
    if soft_t.shape[0] > 0:
        full = torch.cat([full, soft_t.unsqueeze(0).expand(B, -1, -1)], dim=1)
    return full


# ---------------------------------------------------------------------------
# ep_offset builder for orientation batching across macros
# ---------------------------------------------------------------------------

def _orient_ep_batch(
    cache,
    macro_ep: list,   # macro_ep[m] = list of ep indices belonging to macro m
    bm: np.ndarray,   # [B] macro indices
    bo: np.ndarray,   # [B] orientation ids (0=no change, 1-3=transform)
    device: str,
) -> torch.Tensor:
    """
    Build [B, E, 2] ep_offset_override where batch element b applies
    orientation bo[b] to macro bm[b]'s pins.
    """
    B = len(bm)
    E = cache.ep_macro_idx.shape[0]
    ep_off = cache.ep_offset.to(device).unsqueeze(0).expand(B, E, 2).clone()
    for b in range(B):
        o = int(bo[b])
        if o == 0:
            continue
        pins = macro_ep[int(bm[b])]
        if pins:
            pt = torch.tensor(pins, dtype=torch.long, device=device)
            ep_off[b, pt, 0] *= float(_SX[o])
            ep_off[b, pt, 1] *= float(_SY[o])
    return ep_off


# ---------------------------------------------------------------------------
# Greedy LNS re-pack
# ---------------------------------------------------------------------------

def _lns_repack(
    pos: np.ndarray,
    rip: list,
    sizes: np.ndarray,
    hw: np.ndarray,
    hh: np.ndarray,
    cw: float,
    ch: float,
    sx: np.ndarray,
    sy: np.ndarray,
) -> np.ndarray:
    new_pos = pos.copy()
    centroid = new_pos[rip].mean(0)
    old = {i: pos[i].copy() for i in rip}
    order = sorted(rip, key=lambda i: -sizes[i, 0] * sizes[i, 1])
    step = max(max(sizes[i, 0] for i in rip), max(sizes[i, 1] for i in rip)) * 0.5
    for idx in order:
        placed = False
        for r in range(1, 80):
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    nx = np.clip(centroid[0] + dxm * step, hw[idx], cw - hw[idx])
                    ny = np.clip(centroid[1] + dym * step, hh[idx], ch - hh[idx])
                    new_pos[idx] = [nx, ny]
                    if not _overlap(new_pos, idx, sx, sy):
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            new_pos[idx] = old[idx]
    return new_pos


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def coord_descent_polish(
    placement: torch.Tensor,       # [N, 2] from SA
    orientations: torch.Tensor,    # [N] int
    benchmark: Benchmark,
    time_budget_s: float = 300,
    spacing_um: float = 12.0,
    lns_every: int = 4,
    lns_cluster: int = 8,
) -> tuple:
    """
    Polish a placement via coordinate descent + orientation search + LNS.
    Returns (refined_placement [N,2], refined_orientations [N]).
    """
    from fast_proxy_cost import precompute, compute_proxy_cost_gpu
    from submissions.sa_placer.sa_placer import _load_plc

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_hard = benchmark.num_hard_macros
    n_total = benchmark.num_macros
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    canvas_max = max(cw, ch)

    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard].numpy().astype(bool)
    mov_idx = np.where(movable)[0]

    # Auto-disable spacing constraint for dense benchmarks (IBM canvas ~23 μm)
    eff_sp = spacing_um if canvas_max >= 200.0 else 0.0
    if eff_sp == 0.0:
        print(f"  [polish] Canvas {canvas_max:.0f}μm < 200μm → spacing=0 (IBM mode)")

    sx0, sy0 = _sep(sizes, 0.0)       # no-overlap
    sxs, sys = _sep(sizes, eff_sp)    # with spacing

    # ── GPU cache ────────────────────────────────────────────────────────
    plc = _load_plc(benchmark.name)
    if plc is None:
        raise RuntimeError(f"Cannot load plc for '{benchmark.name}'")
    cache = precompute(benchmark, plc, device=device)

    # ── Working state ─────────────────────────────────────────────────────
    pos = placement[:n_hard].numpy().astype(np.float64)
    abs_orients = orientations[:n_hard].numpy().astype(np.int64)

    n_soft = n_total - n_hard
    soft_t = placement[n_hard:].to(device) if n_soft > 0 else torch.zeros(0, 2, device=device)

    # ep-index per hard macro
    macro_ep: list = [[] for _ in range(n_hard)]
    for e in range(cache.ep_macro_idx.shape[0]):
        m = cache.ep_macro_idx[e].item()
        if 0 <= m < n_hard and not cache.ep_is_port[e].item():
            macro_ep[m].append(e)
    has_pins = np.array([len(macro_ep[m]) > 0 for m in range(n_hard)], dtype=bool)

    # ── Helpers ───────────────────────────────────────────────────────────
    def _eval1(p: np.ndarray) -> float:
        pt = torch.tensor(p, dtype=torch.float32, device=device).unsqueeze(0)
        if n_soft > 0:
            pt = torch.cat([pt, soft_t.unsqueeze(0)], dim=1)
        return float(compute_proxy_cost_gpu(pt, cache).item())

    def _eval_b(full: torch.Tensor, ep_off=None) -> np.ndarray:
        return compute_proxy_cost_gpu(full, cache,
                                      ep_offset_override=ep_off).cpu().numpy()

    def _run_batch(bm, bxy, ep_over=None):
        """Evaluate one chunk (bm, bxy arrays). Returns costs array."""
        full = _make_batch(pos, soft_t, bm, bxy, device)
        ep_off = None
        if ep_over is not None:
            ep_off = _orient_ep_batch(cache, macro_ep, bm,
                                      ep_over, device)
        return _eval_b(full, ep_off)

    # ── Initial cost ──────────────────────────────────────────────────────
    ref_cost = _eval1(pos)
    best_cost = ref_cost
    best_pos = pos.copy()
    best_orients = abs_orients.copy()
    print(f"  [polish] Start: {ref_cost:.5f}  spacing={eff_sp:.0f}μm  "
          f"device={device}")

    BATCH = 64
    K_SIDE = 4    # 4×4 = 16 position candidates
    TOL = 1e-5
    schedule = [(0.10, 2), (0.03, 3), (0.01, 999)]
    phase, phase_cnt, pass_idx = 0, 0, 0
    rng = np.random.default_rng(42)

    # ── Main loop ─────────────────────────────────────────────────────────
    while time.time() - t0 < time_budget_s:
        while phase < len(schedule) - 1 and phase_cnt >= schedule[phase][1]:
            phase += 1
            phase_cnt = 0
        scale = schedule[phase][0]
        spread = scale * canvas_max

        mord = rng.permutation(len(mov_idx))
        midx = mov_idx[mord]        # movable macro indices in random order
        n_mov = len(midx)
        offsets = _grid(spread, K_SIDE)  # [K, 2]
        K = len(offsets)

        # ── Phase 1: position CD ─────────────────────────────────────────
        pre_cost = ref_cost

        # Build all candidates vectorised
        base = pos[midx]                                    # [n_mov, 2]
        cands = base[:, None, :] + offsets[None, :, :]     # [n_mov, K, 2]
        _EPS = 1e-4
        cands[:, :, 0] = np.clip(cands[:, :, 0],
                                 hw[midx][:, None] + _EPS, (cw - hw[midx])[:, None] - _EPS)
        cands[:, :, 1] = np.clip(cands[:, :, 1],
                                 hh[midx][:, None] + _EPS, (ch - hh[midx])[:, None] - _EPS)
        bm_all = np.repeat(midx, K)          # [n_mov*K]
        bxy_all = cands.reshape(-1, 2)       # [n_mov*K, 2]
        total = len(bm_all)
        all_costs = np.full(total, np.inf, dtype=np.float32)

        n_eval = 0
        for s in range(0, total, BATCH):
            if time.time() - t0 >= time_budget_s:
                break
            e = min(s + BATCH, total)
            all_costs[s:e] = _run_batch(bm_all[s:e], bxy_all[s:e])
            n_eval = e

        # Best candidate per macro (among evaluated)
        macro_best: dict = {}
        for ci in range(n_eval):
            m = int(bm_all[ci])
            c = float(all_costs[ci])
            if m not in macro_best or c < macro_best[m][0]:
                macro_best[m] = (c, bxy_all[ci, 0], bxy_all[ci, 1])

        # Greedy acceptance vs pre-pass baseline
        pos_improved = False
        for oi in rng.permutation(n_mov).tolist():
            m = int(midx[oi])
            if m not in macro_best:
                continue
            c, nx, ny = macro_best[m]
            if c >= pre_cost - TOL:
                continue
            old = pos[m].copy()
            pos[m] = [nx, ny]
            if _overlap(pos, m, sxs, sys):
                pos[m] = old
                continue
            pos_improved = True

        if pos_improved:
            ref_cost = _eval1(pos)
            if ref_cost < best_cost:
                best_cost = ref_cost
                best_pos = pos.copy()

        # ── Phase 2: orientation search (batched across macros) ───────────
        orient_improved = False
        # Only macros that have pins and are movable
        with_pins = midx[has_pins[midx]]
        n_op = len(with_pins)
        if n_op > 0:
            # For each macro: 4 orient candidates (orient id 0-3)
            # orient 0 = no change (current cache state)
            op_bm = np.repeat(with_pins, 4)          # [n_op*4]
            op_bo = np.tile(np.arange(4), n_op)      # [n_op*4] orient ids
            op_bxy = np.repeat(pos[with_pins], 4, axis=0)  # [n_op*4, 2]
            total_op = len(op_bm)
            oc_all = np.full(total_op, np.inf, dtype=np.float32)

            n_eval_op = 0
            for s in range(0, total_op, BATCH):
                if time.time() - t0 >= time_budget_s:
                    break
                e = min(s + BATCH, total_op)
                full = _make_batch(pos, soft_t, op_bm[s:e], op_bxy[s:e], device)
                ep_off = _orient_ep_batch(cache, macro_ep,
                                          op_bm[s:e], op_bo[s:e], device)
                oc_all[s:e] = _eval_b(full, ep_off)
                n_eval_op = e

            # For each macro, find best orientation (must beat ref_cost)
            orient_pre = ref_cost
            for mi, m in enumerate(with_pins):
                start = mi * 4
                if start + 4 > n_eval_op:
                    break
                oc4 = oc_all[start:start + 4]
                best_o = int(np.argmin(oc4))
                if best_o == 0 or oc4[best_o] >= orient_pre - TOL:
                    continue
                # Accept: mutate cache.ep_offset for this macro
                pins = macro_ep[int(m)]
                if pins:
                    pt = torch.tensor(pins, dtype=torch.long)
                    cache.ep_offset[pt, 0] *= float(_SX[best_o])
                    cache.ep_offset[pt, 1] *= float(_SY[best_o])
                abs_orients[int(m)] = _K4[abs_orients[int(m)]][best_o]
                orient_improved = True
                ref_cost = float(oc4[best_o])
                if ref_cost < best_cost:
                    best_cost = ref_cost
                    best_pos = pos.copy()
                    best_orients = abs_orients.copy()

        # ── Phase 3: LNS ─────────────────────────────────────────────────
        lns_improved = False
        if (lns_every > 0 and pass_idx > 0 and pass_idx % lns_every == 0
                and len(mov_idx) >= lns_cluster
                and time.time() - t0 < time_budget_s):
            anchor = int(rng.choice(mov_idx))
            d = np.hypot(pos[:, 0] - pos[anchor, 0],
                         pos[:, 1] - pos[anchor, 1])
            d[~movable] = 1e9
            cluster = [int(i) for i in np.argsort(d)[:lns_cluster]
                       if movable[i]]
            if cluster:
                new_pos = _lns_repack(pos, cluster, sizes, hw, hh,
                                      cw, ch, sxs, sys)
                lns_cost = _eval1(new_pos)
                if lns_cost < ref_cost - TOL:
                    pos = new_pos
                    ref_cost = lns_cost
                    lns_improved = True
                    if ref_cost < best_cost:
                        best_cost = ref_cost
                        best_pos = pos.copy()

        pass_improved = pos_improved or orient_improved or lns_improved
        elapsed = time.time() - t0
        flags = ("P" if pos_improved else "") + ("O" if orient_improved else "") + ("L" if lns_improved else "")
        print(f"  [polish] pass={pass_idx:3d}  scale={scale:.2f}  "
              f"cost={ref_cost:.5f}  best={best_cost:.5f}  "
              f"[{flags or '·'}]  t={elapsed:.1f}s")

        phase_cnt += 1
        pass_idx += 1

        if not pass_improved and phase == len(schedule) - 1:
            print("  [polish] Converged.")
            break

    # ── Output ────────────────────────────────────────────────────────────
    # Clamp to strictly-interior bounds (float32 edge tolerance = 1e-4 μm)
    _EPS = 1e-4
    for m in range(n_hard):
        best_pos[m, 0] = np.clip(best_pos[m, 0], hw[m] + _EPS, cw - hw[m] - _EPS)
        best_pos[m, 1] = np.clip(best_pos[m, 1], hh[m] + _EPS, ch - hh[m] - _EPS)

    full_pos = placement.clone()
    full_pos[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
    full_orients = orientations.clone()
    full_orients[:n_hard] = torch.tensor(best_orients, dtype=torch.long)

    elapsed = time.time() - t0
    print(f"  [polish] Done {pass_idx} passes / {elapsed:.1f}s  "
          f"best={best_cost:.5f}")
    return full_pos, full_orients
