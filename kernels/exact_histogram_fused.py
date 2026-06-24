import torch

from ._build_notice import load_gefen_cuda_extension

_EXTENSION_MODULE = None


def _load_extension():
    global _EXTENSION_MODULE
    if _EXTENSION_MODULE is not None:
        return _EXTENSION_MODULE

    _EXTENSION_MODULE = load_gefen_cuda_extension(
        "gefen_exact_histogram_fused_ext",
        [
            "exact_histogram_fused_binding.cpp",
            "exact_histogram_fused_kernel.cu",
        ],
    )
    return _EXTENSION_MODULE


def gefen_exact_histogram_cuda(
    grad_flat: torch.Tensor,
    period: int,
    bin_counts: torch.Tensor,
) -> None:
    tensors = {
        "grad_flat": grad_flat,
        "bin_counts": bin_counts,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected {} to be a torch.Tensor.".format(name))
        elif tensor.device.type != "cuda":
            raise ValueError(
                "Expected {} to be on CUDA, got device {}.".format(name, tensor.device)
            )

    if grad_flat.dim() != 1:
        raise ValueError(
            "Expected grad_flat to be 1D, got dim={}".format(grad_flat.dim())
        )
    if bin_counts.dim() != 1:
        raise ValueError(
            "Expected bin_counts to be 1D, got dim={}".format(bin_counts.dim())
        )
    if bin_counts.dtype != torch.int64:
        raise ValueError(
            "Expected bin_counts to be int64, got {}".format(bin_counts.dtype)
        )
    if period <= 0:
        raise ValueError("Expected period to be positive, got {}".format(period))
    if grad_flat.numel() % period != 0:
        raise ValueError(
            "Expected grad_flat.numel()={} to be divisible by period={}.".format(
                grad_flat.numel(),
                period,
            )
        )
    if bin_counts.numel() <= 0 or bin_counts.numel() > 4096:
        raise ValueError(
            "Expected histogram bins in [1, 4096], got {}".format(bin_counts.numel())
        )

    module = _load_extension()
    module.gefen_exact_histogram_cuda(
        grad_flat.contiguous(),
        period,
        bin_counts,
    )
