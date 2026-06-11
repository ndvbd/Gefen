from pathlib import Path

import torch

_EXACT_DP_NUMBA_KERNEL = None


def save_distribution_histogram_pt_from_counts(
    hist: torch.Tensor,
    prefix: str,
    suffix: str,
    num_samples: int,
    value_range=(-1.0, 1.0),
) -> None:

    if prefix is None:
        raise ValueError(
            "Cannot save gradient histogram because SAVE_CODEBOOK_PREFIX is None."
        )
    if hist.dim() != 1:
        raise ValueError("Expected 1D histogram counts, got dim={}.".format(hist.dim()))
    if hist.numel() <= 0:
        raise ValueError("Expected at least one histogram bin.")
    if num_samples < 0:
        raise ValueError(
            "num_samples must be non-negative, got {}.".format(num_samples)
        )

    prefix_path = Path(prefix).expanduser().resolve()
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    dist_pt_path = prefix_path.parent / "{}_{}.pt".format(prefix_path.stem, suffix)

    hist_cpu = hist.detach().cpu().to(torch.float32)
    left, right = value_range
    bin_edges = torch.linspace(
        float(left), float(right), steps=hist_cpu.numel() + 1, dtype=torch.float32
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = float(right - left) / float(hist_cpu.numel())
    torch.save(
        {
            "hist": hist_cpu,
            "bin_edges": bin_edges,
            "bin_centers": bin_centers,
            "bin_width": bin_width,
            "range": (float(left), float(right)),
            "num_bins": hist_cpu.numel(),
            "num_samples": int(num_samples),
        },
        dist_pt_path,
    )
    print("Saved distribution histogram data to {}".format(dist_pt_path))


def _compute_nearest_assignments(
    codebook: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    idx = torch.searchsorted(codebook, values)
    left_idx = (idx - 1).clamp(0, codebook.numel() - 1)
    right_idx = idx.clamp(0, codebook.numel() - 1)
    return torch.where(
        (values - codebook[left_idx]).abs() <= (values - codebook[right_idx]).abs(),
        left_idx,
        right_idx,
    )


def _build_padded_codebook(
    active_centers: torch.Tensor,
    num_codebooks: int,
    force_endpoints: bool,
) -> torch.Tensor:
    codebook = torch.empty(num_codebooks, dtype=torch.float32)
    num_active_centers = int(active_centers.numel())
    codebook[:num_active_centers] = active_centers
    codebook[num_active_centers:] = active_centers[-1]
    if force_endpoints:
        codebook[0] = -1.0
        codebook[-1] = 1.0
    codebook, _ = codebook.sort()
    return codebook


def _get_exact_dp_numba_kernel():
    global _EXACT_DP_NUMBA_KERNEL
    if _EXACT_DP_NUMBA_KERNEL is not None:
        return _EXACT_DP_NUMBA_KERNEL

    import numpy as np
    from numba import njit

    @njit(cache=True)
    def _run_exact_dp_numba(
        prefix_w, prefix_wx, prefix_wx2, num_codebooks, force_endpoints
    ):
        num_points = prefix_w.shape[0] - 1
        dp = np.full((num_points + 1, num_codebooks + 1), np.inf, dtype=np.float64)
        backtrack = np.full((num_points + 1, num_codebooks + 1), -1, dtype=np.int64)
        dp[0, 0] = 0.0

        for clusters in range(1, num_codebooks + 1):
            for end_idx in range(clusters, num_points + 1):
                best_cost = np.inf
                best_start = -1
                for start_idx in range(clusters, end_idx + 1):
                    sum_w = prefix_w[end_idx] - prefix_w[start_idx - 1]
                    sum_wx = prefix_wx[end_idx] - prefix_wx[start_idx - 1]
                    sum_wx2 = prefix_wx2[end_idx] - prefix_wx2[start_idx - 1]
                    if force_endpoints and clusters == 1:
                        fixed_center = -1.0
                        segment_cost = (
                            sum_wx2
                            - 2.0 * fixed_center * sum_wx
                            + fixed_center * fixed_center * sum_w
                        )
                    elif force_endpoints and clusters == num_codebooks:
                        fixed_center = 1.0
                        segment_cost = (
                            sum_wx2
                            - 2.0 * fixed_center * sum_wx
                            + fixed_center * fixed_center * sum_w
                        )
                    else:
                        segment_cost = sum_wx2 - (sum_wx * sum_wx) / sum_w
                    candidate = dp[start_idx - 1, clusters - 1] + segment_cost
                    if candidate < best_cost:
                        best_cost = candidate
                        best_start = start_idx
                dp[end_idx, clusters] = best_cost
                backtrack[end_idx, clusters] = best_start

        codebook = np.empty(num_codebooks, dtype=np.float32)
        end_idx = num_points
        for cluster_idx in range(num_codebooks, 0, -1):
            start_idx = backtrack[end_idx, cluster_idx]
            if force_endpoints and cluster_idx == 1:
                codebook[cluster_idx - 1] = -1.0
            elif force_endpoints and cluster_idx == num_codebooks:
                codebook[cluster_idx - 1] = 1.0
            else:
                sum_w = prefix_w[end_idx] - prefix_w[start_idx - 1]
                sum_wx = prefix_wx[end_idx] - prefix_wx[start_idx - 1]
                codebook[cluster_idx - 1] = sum_wx / sum_w
            end_idx = start_idx - 1
        return codebook

    _EXACT_DP_NUMBA_KERNEL = _run_exact_dp_numba
    return _EXACT_DP_NUMBA_KERNEL


def _run_exact_dp_python(
    prefix_w, prefix_wx, prefix_wx2, num_codebooks, force_endpoints
):
    num_points = len(prefix_w) - 1
    dp = [[float("inf")] * (num_codebooks + 1) for _ in range(num_points + 1)]
    backtrack = [[-1] * (num_codebooks + 1) for _ in range(num_points + 1)]
    dp[0][0] = 0.0

    for clusters in range(1, num_codebooks + 1):
        for end_idx in range(clusters, num_points + 1):
            best_cost = float("inf")
            best_start = -1
            for start_idx in range(clusters, end_idx + 1):
                sum_w = prefix_w[end_idx] - prefix_w[start_idx - 1]
                sum_wx = prefix_wx[end_idx] - prefix_wx[start_idx - 1]
                sum_wx2 = prefix_wx2[end_idx] - prefix_wx2[start_idx - 1]
                if force_endpoints and clusters == 1:
                    fixed_center = -1.0
                    segment_cost = (
                        sum_wx2
                        - 2.0 * fixed_center * sum_wx
                        + fixed_center * fixed_center * sum_w
                    )
                elif force_endpoints and clusters == num_codebooks:
                    fixed_center = 1.0
                    segment_cost = (
                        sum_wx2
                        - 2.0 * fixed_center * sum_wx
                        + fixed_center * fixed_center * sum_w
                    )
                else:
                    segment_cost = sum_wx2 - (sum_wx * sum_wx) / sum_w
                candidate = dp[start_idx - 1][clusters - 1] + segment_cost
                if candidate < best_cost:
                    best_cost = candidate
                    best_start = start_idx
            dp[end_idx][clusters] = best_cost
            backtrack[end_idx][clusters] = best_start

    codebook = torch.empty(num_codebooks, dtype=torch.float32)
    end_idx = num_points
    for cluster_idx in range(num_codebooks, 0, -1):
        start_idx = backtrack[end_idx][cluster_idx]
        if force_endpoints and cluster_idx == 1:
            codebook[cluster_idx - 1] = -1.0
        elif force_endpoints and cluster_idx == num_codebooks:
            codebook[cluster_idx - 1] = 1.0
        else:
            sum_w = prefix_w[end_idx] - prefix_w[start_idx - 1]
            sum_wx = prefix_wx[end_idx] - prefix_wx[start_idx - 1]
            codebook[cluster_idx - 1] = sum_wx / sum_w
        end_idx = start_idx - 1
    return codebook


def exact_dp(
    histogram_stats: dict,
    num_codebooks: int,
    force_endpoints: bool = True,
) -> torch.Tensor:
    active_centers = histogram_stats["active_centers"].detach().cpu().to(torch.float32)
    active_counts = histogram_stats["active_counts"].detach().cpu().to(torch.float32)

    if active_centers.numel() == 0:
        raise ValueError("Cannot run exact DP on an empty histogram.")
    if not torch.all(active_counts > 0):
        raise ValueError(
            "exact DP expects strictly positive counts for all active histogram bins."
        )
    if num_codebooks > active_centers.numel():

        return _build_padded_codebook(active_centers, num_codebooks, force_endpoints)

    num_points = active_centers.numel()
    prefix_w = torch.zeros(num_points + 1, dtype=torch.float64)
    prefix_wx = torch.zeros(num_points + 1, dtype=torch.float64)
    prefix_wx2 = torch.zeros(num_points + 1, dtype=torch.float64)
    weights64 = active_counts.to(torch.float64)
    centers64 = active_centers.to(torch.float64)
    prefix_w[1:] = torch.cumsum(weights64, dim=0)
    prefix_wx[1:] = torch.cumsum(weights64 * centers64, dim=0)
    prefix_wx2[1:] = torch.cumsum(weights64 * centers64.square(), dim=0)

    complexity_upper_bound = num_points * num_points * num_codebooks

    try:
        exact_dp_kernel = _get_exact_dp_numba_kernel()
    except ModuleNotFoundError:
        print("Gefen: pure Python exact-DP fallback (numba is not installed)")
        codebook = _run_exact_dp_python(
            prefix_w.tolist(),
            prefix_wx.tolist(),
            prefix_wx2.tolist(),
            num_codebooks,
            force_endpoints,
        )
    else:
        codebook_np = exact_dp_kernel(
            prefix_w.numpy(),
            prefix_wx.numpy(),
            prefix_wx2.numpy(),
            num_codebooks,
            force_endpoints,
        )
        codebook = torch.from_numpy(codebook_np)

    codebook, _ = codebook.sort()
    return codebook
