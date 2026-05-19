"""
Validation and benchmark tests for fast_proxy_cost.py.

Tests:
  1. Accuracy: max abs diff < 1e-4 on 100 random placements (ibm01 + ariane133)
  2. Speed: batch sizes 1, 16, 64, 256; target ≥20× speedup at batch 64 vs reference
"""

import sys
import time
import math
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from macro_place.loader import load_benchmark_from_dir, load_benchmark
from macro_place.objective import compute_proxy_cost, _set_placement
from fast_proxy_cost import precompute, compute_proxy_cost_gpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IBMDIR = "external/MacroPlacement/Testcases/ICCAD04"
NG45DIR = "external/MacroPlacement/Flows/NanGate45"

TOL = 1e-4          # max abs diff requirement
N_RANDOM = 100      # number of random placements for accuracy test
BATCH_SIZES = [1, 16, 64, 256]
WARMUP = 3
TIMING_REPS = 10


def random_valid_placement(benchmark, n_samples: int, seed: int = 42) -> torch.Tensor:
    """
    Generate n_samples random placements [n_samples, N, 2] within canvas bounds,
    respecting macro sizes so macros stay (mostly) within canvas.
    """
    torch.manual_seed(seed)
    N = benchmark.num_macros
    W, H = benchmark.canvas_width, benchmark.canvas_height
    hw = benchmark.macro_sizes[:, 0] / 2  # [N]
    hh = benchmark.macro_sizes[:, 1] / 2  # [N]

    # x in [hw, W-hw], y in [hh, H-hh]
    x_lo = hw.clamp(0)
    x_hi = (W - hw).clamp(x_lo)
    y_lo = hh.clamp(0)
    y_hi = (H - hh).clamp(y_lo)

    rand = torch.rand(n_samples, N, 2)
    x = x_lo + rand[:, :, 0] * (x_hi - x_lo)
    y = y_lo + rand[:, :, 1] * (y_hi - y_lo)

    # Keep fixed macros at their original positions
    fixed = benchmark.macro_fixed
    if fixed.any():
        x[:, fixed] = benchmark.macro_positions[fixed, 0]
        y[:, fixed] = benchmark.macro_positions[fixed, 1]

    return torch.stack([x, y], dim=-1).float()


def reference_costs(benchmark, plc, placements: torch.Tensor) -> torch.Tensor:
    """Compute reference proxy cost for each placement sequentially."""
    costs = []
    for i in range(placements.shape[0]):
        result = compute_proxy_cost(placements[i], benchmark, plc)
        costs.append(result["proxy_cost"])
    return torch.tensor(costs, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ibm01():
    bench, plc = load_benchmark_from_dir(f"{IBMDIR}/ibm01")
    cache = precompute(bench, plc, device=DEVICE)
    return bench, plc, cache


def _try_load_ariane133():
    """Try to load ariane133; return None if files not present."""
    ng45_bench_dir = f"{NG45DIR}/ariane133/netlist/output_CT_Grouping"
    netlist = f"{ng45_bench_dir}/ariane133.pb.txt"
    plc_file = f"{ng45_bench_dir}/ariane133.plc"
    if not (os.path.exists(netlist) and os.path.exists(plc_file)):
        return None
    bench, plc = load_benchmark(netlist, plc_file)
    cache = precompute(bench, plc, device=DEVICE)
    return bench, plc, cache


@pytest.fixture(scope="module")
def ariane133():
    result = _try_load_ariane133()
    if result is None:
        pytest.skip("ariane133 netlist/plc not found")
    return result


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------

class TestAccuracy:
    def _check_accuracy(self, bench, plc, cache, n_random=N_RANDOM, seed=42):
        placements = random_valid_placement(bench, n_random, seed=seed)  # [N_random, N, 2]
        placements_gpu = placements.to(DEVICE)

        # GPU batch
        with torch.no_grad():
            gpu_costs = compute_proxy_cost_gpu(placements_gpu, cache)
        gpu_costs_cpu = gpu_costs.cpu()

        # Reference (sequential)
        ref_costs = reference_costs(bench, plc, placements)

        abs_diff = (gpu_costs_cpu - ref_costs).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        print(f"\n  {bench.name}: max_abs_diff={max_diff:.2e}  mean_abs_diff={mean_diff:.2e}")
        assert max_diff < TOL, (
            f"{bench.name}: max abs diff {max_diff:.2e} >= {TOL:.1e}\n"
            f"  worst index: {abs_diff.argmax().item()}\n"
            f"  gpu[worst]: {gpu_costs_cpu[abs_diff.argmax()].item():.6f}\n"
            f"  ref[worst]: {ref_costs[abs_diff.argmax()].item():.6f}"
        )

    def test_ibm01(self, ibm01):
        bench, plc, cache = ibm01
        self._check_accuracy(bench, plc, cache)

    def test_ariane133(self, ariane133):
        bench, plc, cache = ariane133
        self._check_accuracy(bench, plc, cache)


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

def _time_gpu(placements_gpu, cache, reps=TIMING_REPS):
    for _ in range(WARMUP):
        compute_proxy_cost_gpu(placements_gpu, cache)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(reps):
        compute_proxy_cost_gpu(placements_gpu, cache)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.perf_counter() - t0) / reps


def _time_ref_single(bench, plc, placement, reps=TIMING_REPS):
    for _ in range(WARMUP):
        compute_proxy_cost(placement, bench, plc)
    t0 = time.perf_counter()
    for _ in range(reps):
        compute_proxy_cost(placement, bench, plc)
    return (time.perf_counter() - t0) / reps


def run_benchmark(bench, plc, cache, label: str):
    """Run timing benchmark and print results table."""
    single_placement = bench.macro_positions.clone()  # [N, 2]

    ref_time = _time_ref_single(bench, plc, single_placement)
    ref_time_ms = ref_time * 1000

    print(f"\n{'='*70}")
    print(f"Benchmark: {label}")
    print(f"Reference (single, CPU): {ref_time_ms:.2f} ms")
    print(f"{'Batch':>8}  {'GPU ms':>8}  {'Speedup':>9}  {'ms/sample':>10}")
    print("-" * 45)

    results = {}
    for B in BATCH_SIZES:
        placements = random_valid_placement(bench, B).to(DEVICE)
        gpu_time = _time_gpu(placements, cache)
        speedup = ref_time / gpu_time
        ms_per = gpu_time * 1000 / B
        print(f"  {B:>6}  {gpu_time*1000:>8.2f}  {speedup:>9.1f}x  {ms_per*1000:>8.3f} µs")
        results[B] = {"gpu_ms": gpu_time * 1000, "speedup": speedup}

    return results


class TestSpeed:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_speedup_ibm01(self, ibm01):
        bench, plc, cache = ibm01
        results = run_benchmark(bench, plc, cache, "ibm01")
        speedup_64 = results[64]["speedup"]
        print(f"\n  ibm01 batch-64 speedup: {speedup_64:.1f}x  (target: ≥20x)")
        assert speedup_64 >= 20, f"Speedup {speedup_64:.1f}x < 20x target"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
    def test_speedup_ariane133(self, ariane133):
        bench, plc, cache = ariane133
        results = run_benchmark(bench, plc, cache, "ariane133")
        speedup_64 = results[64]["speedup"]
        print(f"\n  ariane133 batch-64 speedup: {speedup_64:.1f}x  (target: ≥20x)")
        assert speedup_64 >= 20, f"Speedup {speedup_64:.1f}x < 20x target"


# ---------------------------------------------------------------------------
# Standalone benchmark runner
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    # ibm01
    print("\nLoading ibm01...")
    bench01, plc01 = load_benchmark_from_dir(f"{IBMDIR}/ibm01")
    cache01 = precompute(bench01, plc01, device=DEVICE)
    print(f"  Precompute done: E={cache01.ep_macro_idx.shape[0]}, "
          f"nets={cache01.net_weights.shape[0]}")

    # Quick accuracy check
    placements = random_valid_placement(bench01, 5).to(DEVICE)
    gpu_costs = compute_proxy_cost_gpu(placements, cache01).cpu()
    ref_costs = reference_costs(bench01, plc01, placements.cpu())
    max_diff = (gpu_costs - ref_costs).abs().max().item()
    print(f"  Quick accuracy (5 samples): max_abs_diff={max_diff:.2e}")

    if max_diff >= TOL:
        print(f"  WARNING: max diff {max_diff:.2e} exceeds tolerance {TOL}")
        # Print individual costs for debugging
        for i in range(5):
            print(f"    [{i}] gpu={gpu_costs[i]:.6f}  ref={ref_costs[i]:.6f}  diff={abs(gpu_costs[i]-ref_costs[i]):.2e}")

    # Full accuracy test
    print("\nRunning full accuracy test (100 samples) for ibm01...")
    placements100 = random_valid_placement(bench01, 100).to(DEVICE)
    gpu100 = compute_proxy_cost_gpu(placements100, cache01).cpu()
    ref100 = reference_costs(bench01, plc01, placements100.cpu())
    max_diff100 = (gpu100 - ref100).abs().max().item()
    print(f"  max_abs_diff={max_diff100:.2e}  {'PASS' if max_diff100 < TOL else 'FAIL'}")

    if torch.cuda.is_available():
        results = run_benchmark(bench01, plc01, cache01, "ibm01")

    # ariane133
    ariane_data = _try_load_ariane133()
    if ariane_data is not None:
        bench_a, plc_a, cache_a = ariane_data
        print(f"\nLoading ariane133... done")

        placements_a = random_valid_placement(bench_a, 5).to(DEVICE)
        gpu_a = compute_proxy_cost_gpu(placements_a, cache_a).cpu()
        ref_a = reference_costs(bench_a, plc_a, placements_a.cpu())
        diff_a = (gpu_a - ref_a).abs().max().item()
        print(f"  Quick accuracy (5 samples): max_abs_diff={diff_a:.2e}")

        if torch.cuda.is_available():
            run_benchmark(bench_a, plc_a, cache_a, "ariane133")
    else:
        print("\nariane133 not found, skipping.")

    print("\nDone.")


if __name__ == "__main__":
    main()
