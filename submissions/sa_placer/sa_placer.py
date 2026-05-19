"""
Simulated Annealing macro placer with multi-worker GPU batching and
Go-With-The-Winners (GWTW) synchronization.

Algorithm:
- B independent workers, each holding a candidate hard-macro placement
- 5 move operators: shift, swap, teleport, shuffle, mirror
- AABB overlap rejection before cost evaluation
- Batched GPU proxy cost via fast_proxy_cost.py
- Metropolis acceptance with geometric temperature cooling
- GWTW: every sync_freq steps, clone top-25% workers into bottom-25%
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark

# Ensure project root is on sys.path so fast_proxy_cost can be imported
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Benchmark loading helper
# ---------------------------------------------------------------------------

def _load_plc(name: str):
    from macro_place.loader import load_benchmark_from_dir, load_benchmark
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    ng_name = ng45.get(name)
    if ng_name:
        base = (Path("external/MacroPlacement/Flows/NanGate45")
                / ng_name / "netlist" / "output_CT_Grouping")
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(
                str(base / "netlist.pb.txt"), str(base / "initial.plc")
            )
            return plc
    return None


# ---------------------------------------------------------------------------
# Legalization (greedy spiral — minimum displacement from initial)
# ---------------------------------------------------------------------------

def _legalize(pos: np.ndarray, movable: np.ndarray,
              sizes: np.ndarray, cw: float, ch: float) -> np.ndarray:
    n = len(pos)
    hw = sizes[:, 0] / 2
    hh = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
    placed = np.zeros(n, dtype=bool)
    legal = pos.copy()

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        legal[idx, 0] = np.clip(legal[idx, 0], hw[idx], cw - hw[idx])
        legal[idx, 1] = np.clip(legal[idx, 1], hh[idx], ch - hh[idx])

        if placed.any():
            dx = np.abs(legal[idx, 0] - legal[:, 0])
            dy = np.abs(legal[idx, 1] - legal[:, 1])
            conflict = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
            conflict[idx] = False
            if conflict.any():
                step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
                best_p = legal[idx].copy()
                best_d = float("inf")
                for r in range(1, 200):
                    found = False
                    for dxm in range(-r, r + 1):
                        for dym in range(-r, r + 1):
                            if abs(dxm) != r and abs(dym) != r:
                                continue
                            cx = np.clip(pos[idx, 0] + dxm * step, hw[idx], cw - hw[idx])
                            cy = np.clip(pos[idx, 1] + dym * step, hh[idx], ch - hh[idx])
                            dx2 = np.abs(cx - legal[:, 0])
                            dy2 = np.abs(cy - legal[:, 1])
                            c2 = (dx2 < sep_x[idx] + 0.05) & (dy2 < sep_y[idx] + 0.05) & placed
                            c2[idx] = False
                            if c2.any():
                                continue
                            d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                            if d < best_d:
                                best_d = d
                                best_p = np.array([cx, cy])
                                found = True
                    if found:
                        break
                legal[idx] = best_p

        placed[idx] = True

    return legal


# ---------------------------------------------------------------------------
# AABB overlap check (O(N) per macro)
# ---------------------------------------------------------------------------

def _overlaps(pos: np.ndarray, idx: int,
              sep_x: np.ndarray, sep_y: np.ndarray) -> bool:
    dx = np.abs(pos[:, 0] - pos[idx, 0])
    dy = np.abs(pos[:, 1] - pos[idx, 1])
    ov = (dx < sep_x[idx]) & (dy < sep_y[idx])
    ov[idx] = False
    return bool(ov.any())


# ---------------------------------------------------------------------------
# SAPlacer
# ---------------------------------------------------------------------------

class SAPlacer:
    """
    Multi-worker Simulated Annealing macro placer.

    Parameters
    ----------
    num_workers : int
        Number of parallel SA workers (batch size for GPU evaluation).
    time_budget_s : float
        Wall-clock seconds allowed per benchmark.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self, num_workers: int = 64, time_budget_s: float = 2700,
                 seed: int = 0):
        self.num_workers = num_workers
        self.time_budget_s = time_budget_s
        self.seed = seed
        self.sync_freq = 1000
        self.T_init = 1.0
        self.T_min = 1e-4
        self.T_decay = 0.995

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        t0 = time.time()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        B = self.num_workers

        n_hard = benchmark.num_hard_macros
        n_total = benchmark.num_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)

        sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        hw = sizes[:, 0] / 2
        hh = sizes[:, 1] / 2
        movable = benchmark.get_movable_mask()[:n_hard].numpy()
        mov_idx = np.where(movable)[0]

        if len(mov_idx) == 0:
            return benchmark.macro_positions.clone()

        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2

        plc = _load_plc(benchmark.name)
        if plc is None:
            raise RuntimeError(f"Cannot load plc for benchmark '{benchmark.name}'")

        from fast_proxy_cost import precompute, compute_proxy_cost_gpu
        print(f"  Precomputing GPU cache for {benchmark.name}...")
        cache = precompute(benchmark, plc, device=device)

        # ── Legalize initial hard macro positions ────────────────────────
        init = benchmark.macro_positions[:n_hard].numpy().astype(np.float64)
        legal = _legalize(init, movable, sizes, cw, ch)
        print(f"  Legalized initial placement.")

        # ── Initialize workers with small perturbations ──────────────────
        pos = np.tile(legal, (B, 1, 1))  # [B, n_hard, 2]
        for b in range(1, B):
            sig = max(cw, ch) * 0.03
            for i in mov_idx:
                pos[b, i, 0] = np.clip(
                    pos[b, i, 0] + rng.normal(0, sig), hw[i], cw - hw[i])
                pos[b, i, 1] = np.clip(
                    pos[b, i, 1] + rng.normal(0, sig), hh[i], ch - hh[i])

        # Soft macro positions are fixed during SA
        n_soft = n_total - n_hard
        soft_t = benchmark.macro_positions[n_hard:].to(device)  # [n_soft, 2]

        def _eval(p: np.ndarray) -> np.ndarray:
            pt = torch.tensor(p, dtype=torch.float32, device=device)
            if n_soft > 0:
                full = torch.cat([pt, soft_t.unsqueeze(0).expand(B, -1, -1)], dim=1)
            else:
                full = pt
            return compute_proxy_cost_gpu(full, cache).cpu().numpy()

        costs = _eval(pos)
        best_costs = costs.copy()
        best_pos = pos.copy()

        T = float(self.T_init)
        step = 0

        print(f"  Starting SA: B={B}, T={T}, budget={self.time_budget_s}s")
        print(f"  Initial proxy: mean={costs.mean():.4f} min={costs.min():.4f}")

        # ── Main SA loop ─────────────────────────────────────────────────
        while time.time() - t0 < self.time_budget_s:
            prop = pos.copy()

            for b in range(B):
                move = int(rng.integers(5))
                # Fall back to shift for moves requiring more macros
                if move == 1 and len(mov_idx) < 2:
                    move = 0
                if move == 3 and len(mov_idx) < 3:
                    move = 0

                i = int(mov_idx[rng.integers(len(mov_idx))])

                if move == 0:
                    # Shift: Gaussian displacement
                    sig = max(cw, ch) * T * 0.3
                    old = prop[b, i].copy()
                    prop[b, i, 0] = np.clip(
                        old[0] + rng.normal(0, sig), hw[i], cw - hw[i])
                    prop[b, i, 1] = np.clip(
                        old[1] + rng.normal(0, sig), hh[i], ch - hh[i])
                    if _overlaps(prop[b], i, sep_x, sep_y):
                        prop[b, i] = old

                elif move == 1:
                    # Swap: exchange positions of two different macros
                    candidates = [m for m in mov_idx if m != i]
                    j = int(rng.choice(candidates))
                    oi, oj = prop[b, i].copy(), prop[b, j].copy()
                    prop[b, i, 0] = np.clip(oj[0], hw[i], cw - hw[i])
                    prop[b, i, 1] = np.clip(oj[1], hh[i], ch - hh[i])
                    prop[b, j, 0] = np.clip(oi[0], hw[j], cw - hw[j])
                    prop[b, j, 1] = np.clip(oi[1], hh[j], ch - hh[j])
                    if (_overlaps(prop[b], i, sep_x, sep_y)
                            or _overlaps(prop[b], j, sep_x, sep_y)):
                        prop[b, i] = oi
                        prop[b, j] = oj

                elif move == 2:
                    # Teleport: move to a uniformly random valid position
                    old = prop[b, i].copy()
                    prop[b, i, 0] = rng.uniform(hw[i], cw - hw[i])
                    prop[b, i, 1] = rng.uniform(hh[i], ch - hh[i])
                    if _overlaps(prop[b], i, sep_x, sep_y):
                        prop[b, i] = old

                elif move == 3:
                    # Shuffle: cyclic rotation of k randomly chosen macros
                    k = min(int(rng.integers(3, 6)), len(mov_idx))
                    chosen = rng.choice(mov_idx, size=k, replace=False).tolist()
                    olds = prop[b, chosen].copy()
                    for ki in range(k):
                        src = olds[(ki + 1) % k]
                        ci = chosen[ki]
                        prop[b, ci, 0] = np.clip(src[0], hw[ci], cw - hw[ci])
                        prop[b, ci, 1] = np.clip(src[1], hh[ci], ch - hh[ci])
                    if any(_overlaps(prop[b], ci, sep_x, sep_y) for ci in chosen):
                        for ki, ci in enumerate(chosen):
                            prop[b, ci] = olds[ki]

                else:
                    # Mirror: reflect position across canvas centre
                    old = prop[b, i].copy()
                    prop[b, i, 0] = np.clip(cw - old[0], hw[i], cw - hw[i])
                    prop[b, i, 1] = np.clip(ch - old[1], hh[i], ch - hh[i])
                    if _overlaps(prop[b], i, sep_x, sep_y):
                        prop[b, i] = old

            # ── GPU evaluation ───────────────────────────────────────────
            new_costs = _eval(prop)

            # ── Metropolis acceptance ────────────────────────────────────
            delta = new_costs - costs
            u = rng.random(B)
            accept = (delta < 0) | (np.log(u + 1e-300) < -delta / max(T, 1e-10))

            for b in range(B):
                if accept[b]:
                    pos[b] = prop[b]
                    costs[b] = new_costs[b]
                    if costs[b] < best_costs[b]:
                        best_costs[b] = costs[b]
                        best_pos[b] = pos[b].copy()

            # ── GWTW synchronisation ─────────────────────────────────────
            if step > 0 and step % self.sync_freq == 0:
                ranked = np.argsort(costs)
                k = max(1, B // 4)
                winners = ranked[:k]
                losers = ranked[B - k:]
                sig_gwtw = max(cw, ch) * 0.01
                for li, loser in enumerate(losers):
                    src = int(winners[li % k])
                    pos[loser] = pos[src].copy()
                    costs[loser] = costs[src]
                    best_pos[loser] = best_pos[src].copy()
                    best_costs[loser] = best_costs[src]
                    for ii in mov_idx:
                        pos[loser, ii, 0] = np.clip(
                            pos[loser, ii, 0] + rng.normal(0, sig_gwtw),
                            hw[ii], cw - hw[ii])
                        pos[loser, ii, 1] = np.clip(
                            pos[loser, ii, 1] + rng.normal(0, sig_gwtw),
                            hh[ii], ch - hh[ii])

            # ── Temperature update ───────────────────────────────────────
            T = max(T * self.T_decay, self.T_min)
            step += 1

            if step % 5000 == 0:
                elapsed = time.time() - t0
                print(
                    f"  step={step:6d}  T={T:.5f}  "
                    f"best={best_costs.min():.4f}  avg={costs.mean():.4f}  "
                    f"t={elapsed:.1f}s"
                )

        # ── Return best worker's placement ───────────────────────────────
        best_w = int(np.argmin(best_costs))
        best_hard = best_pos[best_w].copy()

        # Final legalization pass to guarantee zero overlaps
        for _ in range(5):
            if any(_overlaps(best_hard, int(i), sep_x, sep_y) for i in mov_idx):
                best_hard = _legalize(best_hard, movable, sizes, cw, ch)
            else:
                break

        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.tensor(best_hard, dtype=torch.float32)

        elapsed = time.time() - t0
        print(
            f"  SA done: {step} steps in {elapsed:.1f}s  "
            f"best proxy={best_costs.min():.4f}"
        )
        return full
