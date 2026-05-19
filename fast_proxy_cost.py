"""
GPU-accelerated batched proxy cost evaluator.

Replicates the TILOS PlacementCost proxy cost exactly, but processes
B independent placements in parallel on GPU.

Proxy cost = 1.0 * WL_cost + 0.5 * density_cost + 0.5 * congestion_cost
where:
  WL_cost         = pin-HPWL / ((W+H) * net_cnt)
  density_cost    = 0.5 * mean(top-10% grid cell densities)
  congestion_cost = mean(top-5% of [V_cong; H_cong] concatenated)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark
# Ensure monkey-patch is applied before any PlacementCost is used
import macro_place.objective  # noqa: F401


# ---------------------------------------------------------------------------
# Cache dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkGPUCache:
    """Precomputed static tensors for batched GPU evaluation."""

    # Endpoint table (E = total pin entries across all nets, with duplicates)
    ep_macro_idx: torch.Tensor   # [E] int64, index into placements[:,e,:]; -1 for ports
    ep_offset: torch.Tensor      # [E, 2] float32
    ep_fixed_pos: torch.Tensor   # [E, 2] float32 (meaningful only for ports)
    ep_is_port: torch.Tensor     # [E] bool

    # Padded net-endpoint table for WL
    net_ep_table: torch.Tensor   # [num_nets, max_pins] int64  (0-padded)
    net_ep_valid: torch.Tensor   # [num_nets, max_pins] bool
    net_weights: torch.Tensor    # [num_nets] float32

    # Net groups for routing
    net2_ep: torch.Tensor        # [N2, 2] int64
    net2_w: torch.Tensor         # [N2] float32
    net3_ep: torch.Tensor        # [N3, 3] int64
    net3_w: torch.Tensor         # [N3] float32
    netk_ep: torch.Tensor        # [Nk, max_k] int64 (0-padded)
    netk_ep_valid: torch.Tensor  # [Nk, max_k] bool
    netk_w: torch.Tensor         # [Nk] float32
    netk_driver_ep: torch.Tensor # [Nk] int64

    # Hard macro info for macro routing
    hard_macro_bench_idx: torch.Tensor  # [H] int64
    hard_macro_w: torch.Tensor          # [H] float32
    hard_macro_h: torch.Tensor          # [H] float32

    # All macro sizes for density
    all_macro_w: torch.Tensor    # [N] float32
    all_macro_h: torch.Tensor    # [N] float32

    # Canvas / grid scalars
    canvas_width: float
    canvas_height: float
    grid_rows: int
    grid_cols: int
    cell_w: float
    cell_h: float
    grid_v_routes: float
    grid_h_routes: float
    smooth_range: int
    hrouting_alloc: float
    vrouting_alloc: float
    net_cnt: float

    # Smoothing index arrays
    v_smooth_lo: torch.Tensor   # [C]
    v_smooth_hi: torch.Tensor   # [C]
    v_smooth_cnt: torch.Tensor  # [C] float32
    h_smooth_lo: torch.Tensor   # [R]
    h_smooth_hi: torch.Tensor   # [R]
    h_smooth_cnt: torch.Tensor  # [R] float32

    density_cnt: int             # floor(R*C * 0.1), min 1

    device: str = "cuda"


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------

def precompute(benchmark: Benchmark, plc, device: str = "cuda") -> BenchmarkGPUCache:
    """
    Extract static tensors from benchmark + PlacementCost object.
    Call once per benchmark.
    """
    R = benchmark.grid_rows
    C = benchmark.grid_cols
    W = benchmark.canvas_width
    H = benchmark.canvas_height
    cell_w = W / C
    cell_h = H / R
    grid_v_routes = cell_w * benchmark.vroutes_per_micron
    grid_h_routes = cell_h * benchmark.hroutes_per_micron

    smooth_range = int(plc.smooth_range)
    hrouting_alloc = float(plc.hrouting_alloc)
    vrouting_alloc = float(plc.vrouting_alloc)
    net_cnt = float(plc.net_cnt)

    # Reverse map: plc module index → benchmark macro index
    plc_to_bench: dict = {}
    for i, plc_idx in enumerate(benchmark.hard_macro_indices):
        plc_to_bench[plc_idx] = i
    for j, plc_idx in enumerate(benchmark.soft_macro_indices):
        plc_to_bench[plc_idx] = benchmark.num_hard_macros + j

    # Build endpoint table from plc.nets
    ep_macro_idx_list: List[int] = []
    ep_offset_list: List[Tuple[float, float]] = []
    ep_fixed_pos_list: List[Tuple[float, float]] = []
    ep_is_port_list: List[bool] = []

    def _add_ep(pin_name: str) -> int:
        plc_pin_idx = plc.mod_name_to_indices[pin_name]
        mod = plc.modules_w_pins[plc_pin_idx]
        ep = len(ep_macro_idx_list)
        if mod.get_type() == "PORT":
            x, y = mod.get_pos()
            ep_macro_idx_list.append(-1)
            ep_offset_list.append((0.0, 0.0))
            ep_fixed_pos_list.append((x, y))
            ep_is_port_list.append(True)
        else:
            ref_plc_idx = plc.get_ref_node_id(plc_pin_idx)
            bench_idx = plc_to_bench[ref_plc_idx]
            ox, oy = mod.get_offset()
            ep_macro_idx_list.append(bench_idx)
            ep_offset_list.append((ox, oy))
            ep_fixed_pos_list.append((0.0, 0.0))
            ep_is_port_list.append(False)
        return ep

    net_ep_lists: List[List[int]] = []
    net_weight_list: List[float] = []

    for driver_name, sink_names in plc.nets.items():
        driver_plc_idx = plc.mod_name_to_indices[driver_name]
        driver_mod = plc.modules_w_pins[driver_plc_idx]
        weight = float(driver_mod.get_weight())

        eps: List[int] = [_add_ep(driver_name)]
        for sn in sink_names:
            eps.append(_add_ep(sn))
        net_ep_lists.append(eps)
        net_weight_list.append(weight)

    # Convert to tensors
    ep_macro_idx_t = torch.tensor(ep_macro_idx_list, dtype=torch.int64)
    ep_offset_t = torch.tensor(ep_offset_list, dtype=torch.float32)
    ep_fixed_pos_t = torch.tensor(ep_fixed_pos_list, dtype=torch.float32)
    ep_is_port_t = torch.tensor(ep_is_port_list, dtype=torch.bool)

    num_nets = len(net_ep_lists)
    net_weights_t = torch.tensor(net_weight_list, dtype=torch.float32)

    max_pins = max(len(eps) for eps in net_ep_lists)
    net_ep_table = torch.zeros(num_nets, max_pins, dtype=torch.int64)
    net_ep_valid_t = torch.zeros(num_nets, max_pins, dtype=torch.bool)
    for n, eps in enumerate(net_ep_lists):
        k = len(eps)
        net_ep_table[n, :k] = torch.tensor(eps, dtype=torch.int64)
        net_ep_valid_t[n, :k] = True

    net2_ep_list, net2_w_list = [], []
    net3_ep_list, net3_w_list = [], []
    netk_ep_list_raw, netk_w_list, netk_driver_list = [], [], []

    for n, eps in enumerate(net_ep_lists):
        w = net_weight_list[n]
        k = len(eps)
        if k == 2:
            net2_ep_list.append(eps)
            net2_w_list.append(w)
        elif k == 3:
            net3_ep_list.append(eps)
            net3_w_list.append(w)
        elif k > 3:
            netk_ep_list_raw.append(eps)
            netk_w_list.append(w)
            netk_driver_list.append(eps[0])

    net2_ep_t = torch.tensor(net2_ep_list, dtype=torch.int64) if net2_ep_list else torch.zeros(0, 2, dtype=torch.int64)
    net2_w_t = torch.tensor(net2_w_list, dtype=torch.float32) if net2_w_list else torch.zeros(0)
    net3_ep_t = torch.tensor(net3_ep_list, dtype=torch.int64) if net3_ep_list else torch.zeros(0, 3, dtype=torch.int64)
    net3_w_t = torch.tensor(net3_w_list, dtype=torch.float32) if net3_w_list else torch.zeros(0)

    if netk_ep_list_raw:
        max_k = max(len(eps) for eps in netk_ep_list_raw)
        Nk = len(netk_ep_list_raw)
        netk_ep_t = torch.zeros(Nk, max_k, dtype=torch.int64)
        netk_ep_valid_t = torch.zeros(Nk, max_k, dtype=torch.bool)
        for i, eps in enumerate(netk_ep_list_raw):
            kk = len(eps)
            netk_ep_t[i, :kk] = torch.tensor(eps, dtype=torch.int64)
            netk_ep_valid_t[i, :kk] = True
        netk_w_t = torch.tensor(netk_w_list, dtype=torch.float32)
        netk_driver_t = torch.tensor(netk_driver_list, dtype=torch.int64)
    else:
        netk_ep_t = torch.zeros(0, 4, dtype=torch.int64)
        netk_ep_valid_t = torch.zeros(0, 4, dtype=torch.bool)
        netk_w_t = torch.zeros(0)
        netk_driver_t = torch.zeros(0, dtype=torch.int64)

    # Hard macro info
    hard_bench_idx_t = torch.arange(benchmark.num_hard_macros, dtype=torch.int64)
    hard_w_t = benchmark.macro_sizes[:benchmark.num_hard_macros, 0].clone()
    hard_h_t = benchmark.macro_sizes[:benchmark.num_hard_macros, 1].clone()

    # All macro sizes for density
    all_w_t = benchmark.macro_sizes[:, 0].clone()
    all_h_t = benchmark.macro_sizes[:, 1].clone()

    # Smoothing lookup tables
    col_idx = torch.arange(C, dtype=torch.int64)
    v_lo = (col_idx - smooth_range).clamp(0, C - 1)
    v_hi = (col_idx + smooth_range).clamp(0, C - 1)
    v_cnt = (v_hi - v_lo + 1).float()

    row_idx = torch.arange(R, dtype=torch.int64)
    h_lo = (row_idx - smooth_range).clamp(0, R - 1)
    h_hi = (row_idx + smooth_range).clamp(0, R - 1)
    h_cnt = (h_hi - h_lo + 1).float()

    density_cnt = max(1, int(math.floor(R * C * 0.1)))

    return BenchmarkGPUCache(
        ep_macro_idx=ep_macro_idx_t.to(device),
        ep_offset=ep_offset_t.to(device),
        ep_fixed_pos=ep_fixed_pos_t.to(device),
        ep_is_port=ep_is_port_t.to(device),
        net_ep_table=net_ep_table.to(device),
        net_ep_valid=net_ep_valid_t.to(device),
        net_weights=net_weights_t.to(device),
        net2_ep=net2_ep_t.to(device),
        net2_w=net2_w_t.to(device),
        net3_ep=net3_ep_t.to(device),
        net3_w=net3_w_t.to(device),
        netk_ep=netk_ep_t.to(device),
        netk_ep_valid=netk_ep_valid_t.to(device),
        netk_w=netk_w_t.to(device),
        netk_driver_ep=netk_driver_t.to(device),
        hard_macro_bench_idx=hard_bench_idx_t.to(device),
        hard_macro_w=hard_w_t.to(device),
        hard_macro_h=hard_h_t.to(device),
        all_macro_w=all_w_t.to(device),
        all_macro_h=all_h_t.to(device),
        canvas_width=W,
        canvas_height=H,
        grid_rows=R,
        grid_cols=C,
        cell_w=cell_w,
        cell_h=cell_h,
        grid_v_routes=grid_v_routes,
        grid_h_routes=grid_h_routes,
        smooth_range=smooth_range,
        hrouting_alloc=hrouting_alloc,
        vrouting_alloc=vrouting_alloc,
        net_cnt=net_cnt,
        v_smooth_lo=v_lo.to(device),
        v_smooth_hi=v_hi.to(device),
        v_smooth_cnt=v_cnt.to(device),
        h_smooth_lo=h_lo.to(device),
        h_smooth_hi=h_hi.to(device),
        h_smooth_cnt=h_cnt.to(device),
        density_cnt=density_cnt,
        device=device,
    )


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_proxy_cost_gpu(
    placements: torch.Tensor,      # [B, N, 2] float32
    cache: BenchmarkGPUCache,
    return_components: bool = False,
):
    """
    Compute proxy cost for a batch of placements.

    Returns [B] tensor, or dict of [B] tensors when return_components=True.
    """
    ep_pos = _compute_ep_positions(placements, cache)
    wl_cost = _compute_wl_cost(ep_pos, cache)
    density_cost = _compute_density_cost(placements, cache)
    congestion_cost = _compute_congestion_cost(ep_pos, placements, cache)
    total = wl_cost + 0.5 * density_cost + 0.5 * congestion_cost

    if return_components:
        return {
            "proxy_cost": total,
            "wl_cost": wl_cost,
            "density_cost": density_cost,
            "congestion_cost": congestion_cost,
        }
    return total


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_ep_positions(placements: torch.Tensor, cache: BenchmarkGPUCache) -> torch.Tensor:
    """Compute all endpoint positions [B, E, 2]."""
    B = placements.shape[0]
    E = cache.ep_macro_idx.shape[0]
    device = placements.device

    non_port = ~cache.ep_is_port
    np_idx = cache.ep_macro_idx[non_port]
    np_off = cache.ep_offset[non_port]

    ep_pos = torch.empty(B, E, 2, dtype=torch.float32, device=device)
    ep_pos[:, non_port, :] = placements[:, np_idx, :] + np_off
    ep_pos[:, cache.ep_is_port, :] = cache.ep_fixed_pos[cache.ep_is_port].unsqueeze(0)

    return ep_pos


def _compute_wl_cost(ep_pos: torch.Tensor, cache: BenchmarkGPUCache) -> torch.Tensor:
    """WL cost = HPWL / ((W+H) * net_cnt)."""
    idx = cache.net_ep_table.clamp(0)             # [num_nets, max_pins]
    net_pin_pos = ep_pos[:, idx, :]               # [B, num_nets, max_pins, 2]

    valid = cache.net_ep_valid.unsqueeze(0).unsqueeze(-1)  # [1, num_nets, max_pins, 1]
    INF = 1e9
    pos_max = net_pin_pos.masked_fill(~valid, -INF)
    pos_min = net_pin_pos.masked_fill(~valid, INF)

    hpwl = cache.net_weights * (
        (pos_max[..., 0].amax(-1) - pos_min[..., 0].amin(-1))
        + (pos_max[..., 1].amax(-1) - pos_min[..., 1].amin(-1))
    )  # [B, num_nets]

    return hpwl.sum(-1) / ((cache.canvas_width + cache.canvas_height) * cache.net_cnt)


def _compute_density_cost(placements: torch.Tensor, cache: BenchmarkGPUCache) -> torch.Tensor:
    """Density cost = 0.5 * mean(top density_cnt cells / density_cnt)."""
    B = placements.shape[0]
    R, C = cache.grid_rows, cache.grid_cols
    cw, ch = cache.cell_w, cache.cell_h
    device = placements.device

    half_w = cache.all_macro_w * 0.5  # [N]
    half_h = cache.all_macro_h * 0.5

    xl = placements[..., 0] - half_w  # [B, N]
    xr = placements[..., 0] + half_w
    yl = placements[..., 1] - half_h
    yr = placements[..., 1] + half_h

    cx_lo = torch.arange(C, device=device, dtype=torch.float32) * cw
    cx_hi = cx_lo + cw
    cy_lo = torch.arange(R, device=device, dtype=torch.float32) * ch
    cy_hi = cy_lo + ch

    x_ov = (torch.minimum(xr.unsqueeze(-1), cx_hi) - torch.maximum(xl.unsqueeze(-1), cx_lo)).clamp(0)  # [B, N, C]
    y_ov = (torch.minimum(yr.unsqueeze(-1), cy_hi) - torch.maximum(yl.unsqueeze(-1), cy_lo)).clamp(0)  # [B, N, R]

    # density[b,r,c] = sum_n y_ov[b,n,r] * x_ov[b,n,c] / cell_area
    density = torch.bmm(y_ov.permute(0, 2, 1), x_ov) / (cw * ch)  # [B, R, C]

    flat = density.reshape(B, -1)
    sorted_d, _ = flat.sort(dim=-1, descending=True)
    return 0.5 * sorted_d[:, :cache.density_cnt].sum(-1) / cache.density_cnt


def _pos_to_gcell(pos: torch.Tensor, cache: BenchmarkGPUCache) -> torch.Tensor:
    """Convert (x,y) positions to clamped (row, col) grid cell indices."""
    col = (pos[..., 0] / cache.cell_w).floor().long().clamp(0, cache.grid_cols - 1)
    row = (pos[..., 1] / cache.cell_h).floor().long().clamp(0, cache.grid_rows - 1)
    return torch.stack([row, col], dim=-1)


def _add_h_seg(H_diff, b, r, c0, c1, w, R, C1):
    """H_diff[b, r, c0:c1] += w via diff (in-place). c0 < c1 required."""
    valid = c0 < c1
    if not valid.any():
        return
    b_, r_, c0_, c1_, w_ = b[valid], r[valid], c0[valid], c1[valid], w[valid]
    lin_s = b_ * R * C1 + r_ * C1 + c0_
    lin_e = b_ * R * C1 + r_ * C1 + c1_
    flat = H_diff.view(-1)
    flat.scatter_add_(0, lin_s, w_)
    flat.scatter_add_(0, lin_e, -w_)


def _add_v_seg(V_diff, b, r0, r1, c, w, R1, C):
    """V_diff[b, r0:r1, c] += w via diff (in-place). r0 < r1 required."""
    valid = r0 < r1
    if not valid.any():
        return
    b_, r0_, r1_, c_, w_ = b[valid], r0[valid], r1[valid], c[valid], w[valid]
    lin_s = b_ * R1 * C + r0_ * C + c_
    lin_e = b_ * R1 * C + r1_ * C + c_
    flat = V_diff.view(-1)
    flat.scatter_add_(0, lin_s, w_)
    flat.scatter_add_(0, lin_e, -w_)


def _smooth_v(V: torch.Tensor, cache: BenchmarkGPUCache) -> torch.Tensor:
    """Horizontal box-spread of V congestion within ±smooth_range columns."""
    if cache.smooth_range == 0:
        return V
    V_norm = V / cache.v_smooth_cnt.view(1, 1, -1)
    csum = F.pad(V_norm.cumsum(-1), (1, 0))  # [B, R, C+1]
    return csum[:, :, cache.v_smooth_hi + 1] - csum[:, :, cache.v_smooth_lo]


def _smooth_h(H: torch.Tensor, cache: BenchmarkGPUCache) -> torch.Tensor:
    """Vertical box-spread of H congestion within ±smooth_range rows."""
    if cache.smooth_range == 0:
        return H
    H_norm = H / cache.h_smooth_cnt.view(1, -1, 1)
    csum = F.pad(H_norm.cumsum(-2), (0, 0, 1, 0))  # [B, R+1, C]
    return csum[:, cache.h_smooth_hi + 1, :] - csum[:, cache.h_smooth_lo, :]


def _compute_congestion_cost(
    ep_pos: torch.Tensor,
    placements: torch.Tensor,
    cache: BenchmarkGPUCache,
) -> torch.Tensor:
    B = ep_pos.shape[0]
    R, C = cache.grid_rows, cache.grid_cols
    device = ep_pos.device

    ep_gcell = _pos_to_gcell(ep_pos, cache)  # [B, E, 2]

    H_diff = torch.zeros(B, R, C + 1, dtype=torch.float32, device=device)
    V_diff = torch.zeros(B, R + 1, C, dtype=torch.float32, device=device)
    B_range = torch.arange(B, device=device)

    if cache.net2_ep.shape[0] > 0:
        _route_2pin(ep_gcell, cache.net2_ep, cache.net2_w, H_diff, V_diff, B_range, R, C)
    if cache.net3_ep.shape[0] > 0:
        _route_3pin(ep_gcell, cache.net3_ep, cache.net3_w, H_diff, V_diff, B_range, R, C)
    if cache.netk_ep.shape[0] > 0:
        _route_kpin(ep_gcell, cache.netk_ep, cache.netk_ep_valid,
                    cache.netk_w, cache.netk_driver_ep,
                    H_diff, V_diff, B_range, R, C)

    H_net = H_diff[:, :, :C].cumsum(-1) / cache.grid_h_routes
    V_net = V_diff[:, :R, :].cumsum(-2) / cache.grid_v_routes

    V_net = _smooth_v(V_net, cache)
    H_net = _smooth_h(H_net, cache)

    if cache.vrouting_alloc != 0.0 or cache.hrouting_alloc != 0.0:
        V_m, H_m = _macro_routing(placements, cache)
        V_net = V_net + V_m
        H_net = H_net + H_m

    combined = torch.cat([V_net.reshape(B, -1), H_net.reshape(B, -1)], dim=-1)
    cnt = max(1, int(math.floor(2 * R * C * 0.05)))
    top, _ = combined.sort(dim=-1, descending=True)
    return top[:, :cnt].mean(-1)


def _route_2pin(ep_gcell, net2_ep, net2_w, H_diff, V_diff, B_range, R, C):
    B, N2 = ep_gcell.shape[0], net2_ep.shape[0]
    src = ep_gcell[:, net2_ep[:, 0], :]  # [B, N2, 2]
    snk = ep_gcell[:, net2_ep[:, 1], :]

    r_s, c_s = src[..., 0], src[..., 1]
    r_t, c_t = snk[..., 0], snk[..., 1]

    w = net2_w.unsqueeze(0).expand(B, -1)
    b = B_range.unsqueeze(1).expand(B, N2)

    _add_h_seg(H_diff, b.reshape(-1), r_s.reshape(-1),
               torch.minimum(c_s, c_t).reshape(-1), torch.maximum(c_s, c_t).reshape(-1),
               w.reshape(-1), R, C + 1)
    _add_v_seg(V_diff, b.reshape(-1),
               torch.minimum(r_s, r_t).reshape(-1), torch.maximum(r_s, r_t).reshape(-1),
               c_t.reshape(-1), w.reshape(-1), R + 1, C)


def _route_3pin(ep_gcell, net3_ep, net3_w, H_diff, V_diff, B_range, R, C):
    B, N3 = ep_gcell.shape[0], net3_ep.shape[0]
    g = ep_gcell[:, net3_ep.reshape(-1), :].reshape(B, N3, 3, 2)

    r0, c0 = g[:, :, 0, 0], g[:, :, 0, 1]
    r1, c1 = g[:, :, 1, 0], g[:, :, 1, 1]
    r2, c2 = g[:, :, 2, 0], g[:, :, 2, 1]

    same_01 = (r0 == r1) & (c0 == c1)
    same_02 = (r0 == r2) & (c0 == c2)
    same_12 = (r1 == r2) & (c1 == c2)
    all_same = same_01 & same_12

    # 2-unique case: treat as 2-pin
    two_unique = (same_01 | same_02 | same_12) & ~all_same
    tu_snk_r = torch.where(same_01 & ~same_12, r2, r1)
    tu_snk_c = torch.where(same_01 & ~same_12, c2, c1)

    # Sort by (col, row) for 3-pin case analysis
    key_col = g[..., 1] * (R + 1) + g[..., 0]  # sort col first
    order = key_col.argsort(-1)
    gs = g.gather(-2, order.unsqueeze(-1).expand(-1, -1, -1, 2))
    y1, x1 = gs[:, :, 0, 0], gs[:, :, 0, 1]
    y2, x2 = gs[:, :, 1, 0], gs[:, :, 1, 1]
    y3, x3 = gs[:, :, 2, 0], gs[:, :, 2, 1]

    three = ~(same_01 | same_02 | same_12)
    is_L = three & (x1 < x2) & (x2 < x3) & (y2 > torch.minimum(y1, y3)) & (y2 < torch.maximum(y1, y3))
    is_c2 = three & ~is_L & (x2 == x3) & (x1 < x2) & (y1 < torch.minimum(y2, y3))
    is_c3 = three & ~is_L & ~is_c2 & (y2 == y3)
    is_T = three & ~is_L & ~is_c2 & ~is_c3

    # Sort by (row, col) for T-routing
    key_row = g[..., 0] * (C + 1) + g[..., 1]
    ord_r = key_row.argsort(-1)
    gr = g.gather(-2, ord_r.unsqueeze(-1).expand(-1, -1, -1, 2))
    ya, xa = gr[:, :, 0, 0], gr[:, :, 0, 1]
    yb, xb = gr[:, :, 1, 0], gr[:, :, 1, 1]
    yc, xc = gr[:, :, 2, 0], gr[:, :, 2, 1]
    xmin_t = torch.minimum(torch.minimum(xa, xb), xc)
    xmax_t = torch.maximum(torch.maximum(xa, xb), xc)

    w = net3_w.unsqueeze(0).expand(B, -1)
    b = B_range.unsqueeze(1).expand(B, N3)

    def f(t):
        return t.reshape(-1)

    # 2-unique → 2-pin
    if two_unique.any():
        m = two_unique
        _add_h_seg(H_diff, f(b[m]), f(r0[m]),
                   f(torch.minimum(c0, tu_snk_c)[m]), f(torch.maximum(c0, tu_snk_c)[m]),
                   f(w[m]), R, C + 1)
        _add_v_seg(V_diff, f(b[m]),
                   f(torch.minimum(r0, tu_snk_r)[m]), f(torch.maximum(r0, tu_snk_r)[m]),
                   f(tu_snk_c[m]), f(w[m]), R + 1, C)

    # L: H(x1→x2,y1), H(x2→x3,y2), V(y1↔y2,x2), V(y2↔y3,x3)
    if is_L.any():
        m = is_L
        _add_h_seg(H_diff, f(b[m]), f(y1[m]), f(x1[m]), f(x2[m]), f(w[m]), R, C + 1)
        _add_h_seg(H_diff, f(b[m]), f(y2[m]), f(x2[m]), f(x3[m]), f(w[m]), R, C + 1)
        _add_v_seg(V_diff, f(b[m]), f(torch.minimum(y1, y2)[m]), f(torch.maximum(y1, y2)[m]), f(x2[m]), f(w[m]), R + 1, C)
        _add_v_seg(V_diff, f(b[m]), f(torch.minimum(y2, y3)[m]), f(torch.maximum(y2, y3)[m]), f(x3[m]), f(w[m]), R + 1, C)

    # Case2: H(x1→x2,y1), V(y1→max(y2,y3),x2)
    if is_c2.any():
        m = is_c2
        _add_h_seg(H_diff, f(b[m]), f(y1[m]), f(x1[m]), f(x2[m]), f(w[m]), R, C + 1)
        _add_v_seg(V_diff, f(b[m]), f(y1[m]), f(torch.maximum(y2, y3)[m]), f(x2[m]), f(w[m]), R + 1, C)

    # Case3: H(x1→x2,y1), H(x2→x3,y2), V(y1↔y2,x2)
    if is_c3.any():
        m = is_c3
        _add_h_seg(H_diff, f(b[m]), f(y1[m]), f(x1[m]), f(x2[m]), f(w[m]), R, C + 1)
        _add_h_seg(H_diff, f(b[m]), f(y2[m]), f(x2[m]), f(x3[m]), f(w[m]), R, C + 1)
        _add_v_seg(V_diff, f(b[m]), f(torch.minimum(y1, y2)[m]), f(torch.maximum(y1, y2)[m]), f(x2[m]), f(w[m]), R + 1, C)

    # T: H(xmin→xmax,yb), V(ya↔yb,xa), V(yb↔yc,xc)
    if is_T.any():
        m = is_T
        _add_h_seg(H_diff, f(b[m]), f(yb[m]), f(xmin_t[m]), f(xmax_t[m]), f(w[m]), R, C + 1)
        _add_v_seg(V_diff, f(b[m]), f(torch.minimum(ya, yb)[m]), f(torch.maximum(ya, yb)[m]), f(xa[m]), f(w[m]), R + 1, C)
        _add_v_seg(V_diff, f(b[m]), f(torch.minimum(yb, yc)[m]), f(torch.maximum(yb, yc)[m]), f(xc[m]), f(w[m]), R + 1, C)


def _route_kpin(ep_gcell, netk_ep, netk_ep_valid, netk_w, netk_driver_ep,
                H_diff, V_diff, B_range, R, C):
    """
    Route k>3 pin nets, exactly replicating reference behavior.

    Per-batch-element dispatch based on unique gcell count after deduplication:
      0 unique non-source → skip
      1 unique → 2-pin routing
      2 unique → 3-pin routing (L/T)
      >2 unique → star decomp from source to each unique non-source gcell
    """
    B = ep_gcell.shape[0]
    device = ep_gcell.device

    for ni in range(netk_ep.shape[0]):
        w_val = netk_w[ni].item()
        w_tensor = torch.tensor(w_val, dtype=torch.float32, device=device)
        drv_ep = netk_driver_ep[ni].item()
        src = ep_gcell[:, drv_ep, :]  # [B, 2]

        valid_mask = netk_ep_valid[ni]
        ep_inds = netk_ep[ni, valid_mask]

        # Build per-batch-element list of unique non-source gcells
        seen_r = src[:, 0:1].clone()  # [B, 1] starts with source
        seen_c = src[:, 1:2].clone()
        unique_snk_r = []  # list of [B] row tensors
        unique_snk_c = []  # list of [B] col tensors
        unique_is_new = [] # list of [B] bool tensors

        for ki in range(ep_inds.shape[0]):
            if ep_inds[ki].item() == drv_ep:
                continue
            snk = ep_gcell[:, ep_inds[ki].item(), :]
            snk_r, snk_c = snk[:, 0], snk[:, 1]
            same_seen = ((snk_r.unsqueeze(1) == seen_r) & (snk_c.unsqueeze(1) == seen_c)).any(1)
            is_new = ~same_seen  # [B]
            seen_r = torch.cat([seen_r, snk_r.unsqueeze(1)], dim=1)
            seen_c = torch.cat([seen_c, snk_c.unsqueeze(1)], dim=1)
            if is_new.any():
                unique_snk_r.append(snk_r)
                unique_snk_c.append(snk_c)
                unique_is_new.append(is_new)

        if not unique_snk_r:
            continue

        # Per-batch-element unique count [B]
        unique_count = unique_is_new[0].long()
        for m in unique_is_new[1:]:
            unique_count = unique_count + m.long()

        # 1 unique → 2-pin routing
        m2 = unique_count == 1
        if m2.any():
            for idx in range(len(unique_snk_r)):
                sel = m2 & unique_is_new[idx]
                if sel.any():
                    b_sel = B_range[sel]
                    sr_sel = src[sel, 0]; sc_sel = src[sel, 1]
                    tk_r = unique_snk_r[idx][sel]; tk_c = unique_snk_c[idx][sel]
                    M = b_sel.shape[0]; w_exp = w_tensor.expand(M)
                    _add_h_seg(H_diff, b_sel, sr_sel,
                               torch.minimum(sc_sel, tk_c), torch.maximum(sc_sel, tk_c),
                               w_exp, R, C + 1)
                    _add_v_seg(V_diff, b_sel,
                               torch.minimum(sr_sel, tk_r), torch.maximum(sr_sel, tk_r),
                               tk_c, w_exp, R + 1, C)

        # 2 unique → 3-pin routing (L/T)
        m3 = unique_count == 2
        if m3.any():
            # Find the first and second unique gcell for each batch element in m3
            first_r  = torch.zeros(B, dtype=torch.long, device=device)
            first_c  = torch.zeros(B, dtype=torch.long, device=device)
            second_r = torch.zeros(B, dtype=torch.long, device=device)
            second_c = torch.zeros(B, dtype=torch.long, device=device)
            first_found  = torch.zeros(B, dtype=torch.bool, device=device)
            second_found = torch.zeros(B, dtype=torch.bool, device=device)

            for idx in range(len(unique_snk_r)):
                is_this = unique_is_new[idx]
                was_first = first_found.clone()
                upd_first  = m3 & is_this & ~first_found
                first_r    = torch.where(upd_first, unique_snk_r[idx], first_r)
                first_c    = torch.where(upd_first, unique_snk_c[idx], first_c)
                first_found = first_found | upd_first
                upd_second = m3 & is_this & was_first & ~second_found
                second_r   = torch.where(upd_second, unique_snk_r[idx], second_r)
                second_c   = torch.where(upd_second, unique_snk_c[idx], second_c)
                second_found = second_found | upd_second

            if m3.any():
                b3 = B_range[m3]
                g = torch.stack([
                    torch.stack([src[m3, 0], src[m3, 1]], dim=1),
                    torch.stack([first_r[m3],  first_c[m3]],  dim=1),
                    torch.stack([second_r[m3], second_c[m3]], dim=1),
                ], dim=1)  # [M3, 3, 2]
                _route_3pin_batch(g, w_tensor.expand(b3.shape[0]), b3, H_diff, V_diff, R, C)

        # >2 unique → star routing from source to each unique non-source gcell
        mstar = unique_count > 2
        if mstar.any():
            for idx in range(len(unique_snk_r)):
                sel = mstar & unique_is_new[idx]
                if sel.any():
                    b_sel = B_range[sel]
                    sr_sel = src[sel, 0]; sc_sel = src[sel, 1]
                    tk_r = unique_snk_r[idx][sel]; tk_c = unique_snk_c[idx][sel]
                    M = b_sel.shape[0]; w_exp = w_tensor.expand(M)
                    _add_h_seg(H_diff, b_sel, sr_sel,
                               torch.minimum(sc_sel, tk_c), torch.maximum(sc_sel, tk_c),
                               w_exp, R, C + 1)
                    _add_v_seg(V_diff, b_sel,
                               torch.minimum(sr_sel, tk_r), torch.maximum(sr_sel, tk_r),
                               tk_c, w_exp, R + 1, C)


def _route_3pin_batch(g, w, b_ids, H_diff, V_diff, R, C):
    """Route a batch of 3-pin nets. g=[M,3,2], w=[M], b_ids=[M]."""
    M = g.shape[0]
    r0, c0 = g[:, 0, 0], g[:, 0, 1]
    r1, c1 = g[:, 1, 0], g[:, 1, 1]
    r2, c2 = g[:, 2, 0], g[:, 2, 1]

    same_01 = (r0 == r1) & (c0 == c1)
    same_02 = (r0 == r2) & (c0 == c2)
    same_12 = (r1 == r2) & (c1 == c2)
    all_same = same_01 & same_12

    two_unique = (same_01 | same_02 | same_12) & ~all_same
    tu_snk_r = torch.where(same_01 & ~same_12, r2, r1)
    tu_snk_c = torch.where(same_01 & ~same_12, c2, c1)

    key_col = g[..., 1] * (R + 1) + g[..., 0]
    order = key_col.argsort(-1)
    gs = g.gather(-2, order.unsqueeze(-1).expand(-1, -1, 2))
    y1, x1 = gs[:, 0, 0], gs[:, 0, 1]
    y2, x2 = gs[:, 1, 0], gs[:, 1, 1]
    y3, x3 = gs[:, 2, 0], gs[:, 2, 1]

    three = ~(same_01 | same_02 | same_12)
    is_L  = three & (x1 < x2) & (x2 < x3) & (y2 > torch.minimum(y1, y3)) & (y2 < torch.maximum(y1, y3))
    is_c2 = three & ~is_L & (x2 == x3) & (x1 < x2) & (y1 < torch.minimum(y2, y3))
    is_c3 = three & ~is_L & ~is_c2 & (y2 == y3)
    is_T  = three & ~is_L & ~is_c2 & ~is_c3

    key_row = g[..., 0] * (C + 1) + g[..., 1]
    ord_r = key_row.argsort(-1)
    gr = g.gather(-2, ord_r.unsqueeze(-1).expand(-1, -1, 2))
    ya, xa = gr[:, 0, 0], gr[:, 0, 1]
    yb, xb = gr[:, 1, 0], gr[:, 1, 1]
    yc, xc = gr[:, 2, 0], gr[:, 2, 1]
    xmin_t = torch.minimum(torch.minimum(xa, xb), xc)
    xmax_t = torch.maximum(torch.maximum(xa, xb), xc)

    def f(t): return t.reshape(-1)

    if two_unique.any():
        m = two_unique
        bm = b_ids[m]; wm = w[m]
        _add_h_seg(H_diff, bm, f(r0[m]),
                   f(torch.minimum(c0, tu_snk_c)[m]), f(torch.maximum(c0, tu_snk_c)[m]),
                   wm, R, C + 1)
        _add_v_seg(V_diff, bm,
                   f(torch.minimum(r0, tu_snk_r)[m]), f(torch.maximum(r0, tu_snk_r)[m]),
                   f(tu_snk_c[m]), wm, R + 1, C)
    if is_L.any():
        m = is_L; bm = b_ids[m]; wm = w[m]
        _add_h_seg(H_diff, bm, f(y1[m]), f(x1[m]), f(x2[m]), wm, R, C+1)
        _add_h_seg(H_diff, bm, f(y2[m]), f(x2[m]), f(x3[m]), wm, R, C+1)
        _add_v_seg(V_diff, bm, f(torch.minimum(y1,y2)[m]), f(torch.maximum(y1,y2)[m]), f(x2[m]), wm, R+1, C)
        _add_v_seg(V_diff, bm, f(torch.minimum(y2,y3)[m]), f(torch.maximum(y2,y3)[m]), f(x3[m]), wm, R+1, C)
    if is_c2.any():
        m = is_c2; bm = b_ids[m]; wm = w[m]
        _add_h_seg(H_diff, bm, f(y1[m]), f(x1[m]), f(x2[m]), wm, R, C+1)
        _add_v_seg(V_diff, bm, f(y1[m]), f(torch.maximum(y2,y3)[m]), f(x2[m]), wm, R+1, C)
    if is_c3.any():
        m = is_c3; bm = b_ids[m]; wm = w[m]
        _add_h_seg(H_diff, bm, f(y1[m]), f(x1[m]), f(x2[m]), wm, R, C+1)
        _add_h_seg(H_diff, bm, f(y2[m]), f(x2[m]), f(x3[m]), wm, R, C+1)
        _add_v_seg(V_diff, bm, f(torch.minimum(y1,y2)[m]), f(torch.maximum(y1,y2)[m]), f(x2[m]), wm, R+1, C)
    if is_T.any():
        m = is_T; bm = b_ids[m]; wm = w[m]
        _add_h_seg(H_diff, bm, f(yb[m]), f(xmin_t[m]), f(xmax_t[m]), wm, R, C+1)
        _add_v_seg(V_diff, bm, f(torch.minimum(ya,yb)[m]), f(torch.maximum(ya,yb)[m]), f(xa[m]), wm, R+1, C)
        _add_v_seg(V_diff, bm, f(torch.minimum(yb,yc)[m]), f(torch.maximum(yb,yc)[m]), f(xc[m]), wm, R+1, C)


def _macro_routing(placements: torch.Tensor, cache: BenchmarkGPUCache):
    """
    Compute normalized V/H macro routing congestion grids [B, R, C].

    Replicates __macro_route_over_grid_cell with partial-row/col corrections.
    """
    B = placements.shape[0]
    R, C = cache.grid_rows, cache.grid_cols
    cw, ch = cache.cell_w, cache.cell_h
    device = placements.device

    hard_idx = cache.hard_macro_bench_idx   # [H]
    hw = cache.hard_macro_w                  # [H]
    hh = cache.hard_macro_h

    pos = placements[:, hard_idx, :]         # [B, H, 2]
    xl = pos[:, :, 0] - hw / 2
    xr = pos[:, :, 0] + hw / 2
    yl = pos[:, :, 1] - hh / 2
    yr = pos[:, :, 1] + hh / 2

    cx_lo = torch.arange(C, device=device, dtype=torch.float32) * cw
    cx_hi = cx_lo + cw
    cy_lo = torch.arange(R, device=device, dtype=torch.float32) * ch
    cy_hi = cy_lo + ch

    # x overlap [B, H, C], y overlap [B, H, R]
    _EPS = 1e-6  # threshold to eliminate floating-point artifacts at cell boundaries
    x_ov = (torch.minimum(xr.unsqueeze(-1), cx_hi) - torch.maximum(xl.unsqueeze(-1), cx_lo)).clamp(0)
    y_ov = (torch.minimum(yr.unsqueeze(-1), cy_hi) - torch.maximum(yl.unsqueeze(-1), cy_lo)).clamp(0)
    x_ov = x_ov.masked_fill(x_ov < _EPS, 0.0)
    y_ov = y_ov.masked_fill(y_ov < _EPS, 0.0)

    # V_macro[b,r,c] = sum_h x_ov[b,h,c] * (y_ov[b,h,r] > 0) * vrouting_alloc
    # H_macro[b,r,c] = sum_h y_ov[b,h,r] * (x_ov[b,h,c] > 0) * hrouting_alloc
    y_in = (y_ov > 0).float()
    x_in = (x_ov > 0).float()
    V_m = torch.bmm(y_in.permute(0, 2, 1), x_ov) * cache.vrouting_alloc   # [B, R, C]
    H_m = torch.bmm(y_ov.permute(0, 2, 1), x_in) * cache.hrouting_alloc   # [B, R, C]

    H_num = hard_idx.shape[0]

    # Partial-row correction for V: if multi-row macro and either boundary row is partial,
    # subtract the top row (ur_row) contribution entirely (matches reference behavior).
    if cache.vrouting_alloc != 0.0 and H_num > 0:
        ur_row = ((yr - 1e-9) / ch).floor().long().clamp(0, R - 1)  # [B, H]
        bl_row = (yl / ch).floor().long().clamp(0, R - 1)
        multi_row = ur_row != bl_row  # [B, H]

        b_flat = torch.arange(B, device=device).unsqueeze(1).expand(B, H_num).reshape(-1)
        h_flat = torch.arange(H_num, device=device).unsqueeze(0).expand(B, H_num).reshape(-1)
        ur_flat = ur_row.reshape(-1)
        bl_flat = bl_row.reshape(-1)
        top_y    = y_ov[b_flat, h_flat, ur_flat].reshape(B, H_num)  # [B, H]
        bottom_y = y_ov[b_flat, h_flat, bl_flat].reshape(B, H_num)  # [B, H]
        # Fire when either boundary row has partial y coverage
        partial_v = multi_row & (((ch - top_y).abs() > 1e-5) | ((ch - bottom_y).abs() > 1e-5))

        if partial_v.any():
            bh = partial_v.nonzero(as_tuple=False)  # [M, 2]
            b_ids, h_ids = bh[:, 0], bh[:, 1]
            r_ids = ur_row[b_ids, h_ids]
            x_contrib = x_ov[b_ids, h_ids, :] * cache.vrouting_alloc  # [M, C]
            lin = b_ids * R * C + r_ids * C  # [M]
            for ci in range(C):
                V_m.view(-1).scatter_add_(0, lin + ci, -x_contrib[:, ci])

    # Partial-col correction for H: if multi-col macro and either boundary col is partial,
    # subtract the right col (ur_col) contribution; only when macro actually overlaps ur_col.
    if cache.hrouting_alloc != 0.0 and H_num > 0:
        ur_col = ((xr - 1e-9) / cw).floor().long().clamp(0, C - 1)
        bl_col = (xl / cw).floor().long().clamp(0, C - 1)
        multi_col = ur_col != bl_col

        b_flat2 = torch.arange(B, device=device).unsqueeze(1).expand(B, H_num).reshape(-1)
        h_flat2 = torch.arange(H_num, device=device).unsqueeze(0).expand(B, H_num).reshape(-1)
        uc_flat  = ur_col.reshape(-1)
        bc_flat  = bl_col.reshape(-1)
        right_x = x_ov[b_flat2, h_flat2, uc_flat].reshape(B, H_num)  # [B, H]
        left_x  = x_ov[b_flat2, h_flat2, bc_flat].reshape(B, H_num)  # [B, H]
        # Fire when either boundary col is partial (including when right_x==0 means macro
        # just touched the boundary but doesn't overlap ur_col — reference also fires then)
        partial_h = multi_col & (((cw - right_x).abs() > 1e-5) | ((cw - left_x).abs() > 1e-5))
        # Only subtract at ur_col when macro actually overlaps that column
        partial_h = partial_h & (right_x > _EPS)

        if partial_h.any():
            bh = partial_h.nonzero(as_tuple=False)
            b_ids, h_ids = bh[:, 0], bh[:, 1]
            c_ids = ur_col[b_ids, h_ids]
            y_contrib = y_ov[b_ids, h_ids, :] * cache.hrouting_alloc  # [M, R]
            lin = b_ids * R * C + c_ids
            for ri in range(R):
                H_m.view(-1).scatter_add_(0, lin + ri * C, -y_contrib[:, ri])

    return V_m / cache.grid_v_routes, H_m / cache.grid_h_routes
