import torch

from .gefen_cuda import load_extension


def _load_extension():
    return load_extension()


def gefen_lloyd_accumulate_cuda(
    grad_flat: torch.Tensor,
    codebook: torch.Tensor,
    period: int,
    sums: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    tensors = {
        "grad_flat": grad_flat,
        "codebook": codebook,
        "sums": sums,
        "counts": counts,
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
    elif codebook.dim() != 1:
        raise ValueError(
            "Expected codebook to be 1D, got dim={}".format(codebook.dim())
        )
    elif sums.dim() != 1 or sums.numel() != codebook.numel():
        raise ValueError("Expected sums to be 1D with one entry per codebook value.")
    elif counts.dim() != 1 or counts.numel() != codebook.numel():
        raise ValueError("Expected counts to be 1D with one entry per codebook value.")
    elif codebook.dtype != torch.float32:
        raise ValueError(
            "Expected codebook to be float32, got {}".format(codebook.dtype)
        )
    elif sums.dtype != torch.float32:
        raise ValueError("Expected sums to be float32, got {}".format(sums.dtype))
    elif counts.dtype != torch.int64:
        raise ValueError("Expected counts to be int64, got {}".format(counts.dtype))
    elif period <= 0:
        raise ValueError("Expected period to be positive, got {}".format(period))
    elif grad_flat.numel() % period != 0:
        raise ValueError(
            "Expected grad_flat.numel()={} to be divisible by period={}.".format(
                grad_flat.numel(),
                period,
            )
        )
    elif codebook.numel() > 256:
        raise ValueError(
            "Expected codebook to have at most 256 entries, got {}".format(
                codebook.numel()
            )
        )

    module = _load_extension()
    module.gefen_lloyd_accumulate_cuda(
        grad_flat.contiguous(),
        codebook.contiguous(),
        period,
        sums,
        counts,
    )


def gefen_lloyd_mse_cuda(
    grad_flat: torch.Tensor,
    old_codebook: torch.Tensor,
    new_codebook: torch.Tensor,
    period: int,
    mse_normalized_sum: torch.Tensor,
    mse_original_sum: torch.Tensor,
) -> None:
    tensors = {
        "grad_flat": grad_flat,
        "old_codebook": old_codebook,
        "new_codebook": new_codebook,
        "mse_normalized_sum": mse_normalized_sum,
        "mse_original_sum": mse_original_sum,
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
    elif old_codebook.dim() != 1 or new_codebook.dim() != 1:
        raise ValueError("Expected old_codebook and new_codebook to be 1D.")
    elif old_codebook.numel() != new_codebook.numel():
        raise ValueError(
            "Expected old_codebook and new_codebook to have the same number of entries."
        )
    elif mse_normalized_sum.numel() != 1 or mse_original_sum.numel() != 1:
        raise ValueError("Expected mse accumulators to be scalar tensors.")
    elif old_codebook.dtype != torch.float32 or new_codebook.dtype != torch.float32:
        raise ValueError("Expected codebooks to be float32.")
    elif (
        mse_normalized_sum.dtype != torch.float32
        or mse_original_sum.dtype != torch.float32
    ):
        raise ValueError("Expected mse accumulators to be float32.")
    elif period <= 0:
        raise ValueError("Expected period to be positive, got {}".format(period))
    elif grad_flat.numel() % period != 0:
        raise ValueError(
            "Expected grad_flat.numel()={} to be divisible by period={}.".format(
                grad_flat.numel(),
                period,
            )
        )
    elif old_codebook.numel() > 256:
        raise ValueError(
            "Expected codebook to have at most 256 entries, got {}".format(
                old_codebook.numel()
            )
        )

    module = _load_extension()
    module.gefen_lloyd_mse_cuda(
        grad_flat.contiguous(),
        old_codebook.contiguous(),
        new_codebook.contiguous(),
        period,
        mse_normalized_sum,
        mse_original_sum,
    )
