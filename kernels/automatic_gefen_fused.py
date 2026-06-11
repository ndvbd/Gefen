import torch

from .gefen_cuda import load_extension


def _load_extension():
    return load_extension()


def automatic_gefen_fused_update_cuda(
    p: torch.Tensor,
    grad_view: torch.Tensor,
    m_sign: torch.Tensor,
    m_magnitude: torch.Tensor,
    stepsize: torch.Tensor,
    codebook: torch.Tensor,
    packed_indices: bool,
    beta1: float,
    lr: float,
) -> None:
    tensors = {
        "p": p,
        "grad_view": grad_view,
        "m_sign": m_sign,
        "m_magnitude": m_magnitude,
        "stepsize": stepsize,
        "codebook": codebook,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected {} to be a torch.Tensor.".format(name))
        elif tensor.device.type != "cuda":
            raise ValueError(
                "Expected {} to be on CUDA, got device {}.".format(name, tensor.device)
            )

    if grad_view.dim() != 2:
        raise ValueError(
            "Expected grad_view to be 2D, got dim={}".format(grad_view.dim())
        )
    elif grad_view.dtype != p.dtype:
        raise ValueError(
            "Expected grad_view dtype {} to match p dtype {}.".format(
                grad_view.dtype, p.dtype
            )
        )
    elif m_magnitude.dim() != 2 or m_magnitude.shape[1] != 1:
        raise ValueError(
            "Expected m_magnitude to have shape [num_blocks, 1], got {}".format(
                tuple(m_magnitude.shape)
            )
        )
    elif stepsize.dim() != 2 or stepsize.shape[1] != 1:
        raise ValueError(
            "Expected stepsize to have shape [num_blocks, 1], got {}".format(
                tuple(stepsize.shape)
            )
        )
    elif grad_view.shape[0] != m_magnitude.shape[0]:
        raise ValueError(
            "Expected grad_view and m_magnitude to have the same number of blocks."
        )
    elif grad_view.shape[0] != stepsize.shape[0]:
        raise ValueError(
            "Expected grad_view and stepsize to have the same number of blocks."
        )
    elif m_sign.dtype != torch.uint8:
        raise ValueError("Expected m_sign to be uint8, got {}".format(m_sign.dtype))
    elif m_magnitude.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            "Expected m_magnitude to be float32, float16, or bfloat16, got {}".format(
                m_magnitude.dtype
            )
        )
    elif stepsize.dtype != torch.float32:
        raise ValueError(
            "Expected stepsize to be float32, got {}".format(stepsize.dtype)
        )
    elif codebook.dtype != torch.float32:
        raise ValueError(
            "Expected codebook to be float32, got {}".format(codebook.dtype)
        )
    elif codebook.numel() > 256:
        raise ValueError(
            "Expected codebook to have at most 256 entries, got {}".format(
                codebook.numel()
            )
        )
    elif packed_indices and m_sign.numel() != (grad_view.numel() + 1) // 2:
        raise ValueError(
            "Expected packed m_sign.numel()={} for grad_view.numel()={}, got {}.".format(
                (grad_view.numel() + 1) // 2,
                grad_view.numel(),
                m_sign.numel(),
            )
        )
    elif not packed_indices and m_sign.numel() != grad_view.numel():
        raise ValueError(
            "Expected unpacked m_sign.numel()={} for grad_view.numel()={}, got {}.".format(
                grad_view.numel(),
                grad_view.numel(),
                m_sign.numel(),
            )
        )

    module = _load_extension()
    module.automatic_gefen_fused_update_cuda(
        p,
        grad_view.contiguous(),
        m_sign,
        m_magnitude,
        stepsize.contiguous(),
        codebook.contiguous(),
        packed_indices,
        beta1,
        lr,
    )


def automatic_gefen_fused_update_from_vmean_cuda(
    p: torch.Tensor,
    grad_view: torch.Tensor,
    m_sign: torch.Tensor,
    m_magnitude: torch.Tensor,
    vmean: torch.Tensor,
    codebook: torch.Tensor,
    packed_indices: bool,
    beta1: float,
    lr: float,
    bias_correction_1: float,
    bias_correction_2: float,
    eps: float,
) -> None:
    tensors = {
        "p": p,
        "grad_view": grad_view,
        "m_sign": m_sign,
        "m_magnitude": m_magnitude,
        "vmean": vmean,
        "codebook": codebook,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected {} to be a torch.Tensor.".format(name))
        elif tensor.device.type != "cuda":
            raise ValueError(
                "Expected {} to be on CUDA, got device {}.".format(name, tensor.device)
            )

    if grad_view.dim() != 2:
        raise ValueError(
            "Expected grad_view to be 2D, got dim={}".format(grad_view.dim())
        )
    elif grad_view.dtype != p.dtype:
        raise ValueError(
            "Expected grad_view dtype {} to match p dtype {}.".format(
                grad_view.dtype, p.dtype
            )
        )
    elif m_magnitude.dim() != 2 or m_magnitude.shape[1] != 1:
        raise ValueError(
            "Expected m_magnitude to have shape [num_blocks, 1], got {}".format(
                tuple(m_magnitude.shape)
            )
        )
    elif vmean.dim() != 2 or vmean.shape[1] != 1:
        raise ValueError(
            "Expected vmean to have shape [num_blocks, 1], got {}".format(
                tuple(vmean.shape)
            )
        )
    elif grad_view.shape[0] != m_magnitude.shape[0]:
        raise ValueError(
            "Expected grad_view and m_magnitude to have the same number of blocks."
        )
    elif grad_view.shape[0] != vmean.shape[0]:
        raise ValueError(
            "Expected grad_view and vmean to have the same number of blocks."
        )
    elif m_sign.dtype != torch.uint8:
        raise ValueError("Expected m_sign to be uint8, got {}".format(m_sign.dtype))
    elif m_magnitude.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            "Expected m_magnitude to be float32, float16, or bfloat16, got {}".format(
                m_magnitude.dtype
            )
        )
    elif vmean.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            "Expected vmean to be float32, float16, or bfloat16, got {}".format(
                vmean.dtype
            )
        )
    elif codebook.dtype != torch.float32:
        raise ValueError(
            "Expected codebook to be float32, got {}".format(codebook.dtype)
        )
    elif codebook.numel() > 256:
        raise ValueError(
            "Expected codebook to have at most 256 entries, got {}".format(
                codebook.numel()
            )
        )
    elif packed_indices and m_sign.numel() != (grad_view.numel() + 1) // 2:
        raise ValueError(
            "Expected packed m_sign.numel()={} for grad_view.numel()={}, got {}.".format(
                (grad_view.numel() + 1) // 2,
                grad_view.numel(),
                m_sign.numel(),
            )
        )
    elif not packed_indices and m_sign.numel() != grad_view.numel():
        raise ValueError(
            "Expected unpacked m_sign.numel()={} for grad_view.numel()={}, got {}.".format(
                grad_view.numel(),
                grad_view.numel(),
                m_sign.numel(),
            )
        )
    elif bias_correction_1 <= 0.0:
        raise ValueError(
            "Expected bias_correction_1 to be positive, got {}".format(
                bias_correction_1
            )
        )
    elif bias_correction_2 <= 0.0:
        raise ValueError(
            "Expected bias_correction_2 to be positive, got {}".format(
                bias_correction_2
            )
        )
    elif eps < 0.0:
        raise ValueError("Expected eps to be non-negative, got {}".format(eps))

    module = _load_extension()
    module.automatic_gefen_fused_update_from_vmean_cuda(
        p,
        grad_view.contiguous(),
        m_sign,
        m_magnitude,
        vmean.contiguous(),
        codebook.contiguous(),
        packed_indices,
        beta1,
        lr,
        bias_correction_1,
        bias_correction_2,
        eps,
    )


def automatic_gefen_period1_update_cuda(
    p: torch.Tensor,
    grad_flat: torch.Tensor,
    m_sign: torch.Tensor,
    m_magnitude: torch.Tensor,
    vmean: torch.Tensor,
    codebook: torch.Tensor,
    period_one_divisor: int,
    beta1: float,
    beta2: float,
    lr: float,
    bias_correction_1: float,
    bias_correction_2: float,
    eps: float,
) -> None:
    tensors = {
        "p": p,
        "grad_flat": grad_flat,
        "m_sign": m_sign,
        "m_magnitude": m_magnitude,
        "vmean": vmean,
        "codebook": codebook,
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
    elif grad_flat.dtype != p.dtype:
        raise ValueError(
            "Expected grad_flat dtype {} to match p dtype {}.".format(
                grad_flat.dtype, p.dtype
            )
        )
    elif p.numel() != grad_flat.numel():
        raise ValueError("Expected p.numel() to match grad_flat.numel().")
    elif m_sign.dtype != torch.uint8:
        raise ValueError("Expected m_sign to be uint8, got {}".format(m_sign.dtype))
    elif m_sign.numel() != grad_flat.numel():
        raise ValueError("Expected m_sign.numel() to match grad_flat.numel().")
    elif m_magnitude.dim() != 2 or m_magnitude.shape[1] != 1:
        raise ValueError(
            "Expected m_magnitude to have shape [num_magnitude_blocks, 1], got {}".format(
                tuple(m_magnitude.shape)
            )
        )
    elif vmean.dim() != 2 or vmean.shape[1] != 1:
        raise ValueError(
            "Expected vmean to have shape [numel, 1], got {}".format(tuple(vmean.shape))
        )
    elif not isinstance(period_one_divisor, int):
        raise TypeError(
            "Expected period_one_divisor to be an int, got {}".format(
                type(period_one_divisor).__name__
            )
        )
    elif period_one_divisor <= 0:
        raise ValueError(
            "Expected period_one_divisor to be positive, got {}".format(
                period_one_divisor
            )
        )
    elif grad_flat.numel() % period_one_divisor != 0:
        raise ValueError(
            "Expected period_one_divisor {} to divide grad_flat.numel() {}.".format(
                period_one_divisor,
                grad_flat.numel(),
            )
        )
    elif m_magnitude.shape[0] != grad_flat.numel() // period_one_divisor:
        raise ValueError(
            "Expected m_magnitude to have one row per period-one magnitude block."
        )
    elif vmean.shape[0] != grad_flat.numel():
        raise ValueError("Expected vmean to have one row per gradient element.")
    elif m_magnitude.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            "Expected m_magnitude to be float32, float16, or bfloat16, got {}".format(
                m_magnitude.dtype
            )
        )
    elif vmean.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            "Expected vmean to be float32, float16, or bfloat16, got {}".format(
                vmean.dtype
            )
        )
    elif codebook.dtype != torch.float32:
        raise ValueError(
            "Expected codebook to be float32, got {}".format(codebook.dtype)
        )
    elif codebook.numel() > 256:
        raise ValueError(
            "Expected codebook to have at most 256 entries, got {}".format(
                codebook.numel()
            )
        )
    elif not 0.0 <= beta2 <= 1.0:
        raise ValueError("Expected beta2 to be in [0, 1], got {}".format(beta2))
    elif bias_correction_1 <= 0.0:
        raise ValueError(
            "Expected bias_correction_1 to be positive, got {}".format(
                bias_correction_1
            )
        )
    elif bias_correction_2 <= 0.0:
        raise ValueError(
            "Expected bias_correction_2 to be positive, got {}".format(
                bias_correction_2
            )
        )
    elif eps < 0.0:
        raise ValueError("Expected eps to be non-negative, got {}".format(eps))

    module = _load_extension()
    module.automatic_gefen_period1_update_cuda(
        p,
        grad_flat.contiguous(),
        m_sign,
        m_magnitude,
        vmean,
        codebook.contiguous(),
        period_one_divisor,
        beta1,
        beta2,
        lr,
        bias_correction_1,
        bias_correction_2,
        eps,
    )
