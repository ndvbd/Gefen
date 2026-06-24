import torch

from ._build_notice import load_gefen_cuda_extension

_EXTENSION_MODULE = None


def _load_extension():
    global _EXTENSION_MODULE
    if _EXTENSION_MODULE is not None:
        return _EXTENSION_MODULE

    _EXTENSION_MODULE = load_gefen_cuda_extension(
        "gefen_automatic_fused_ext",
        [
            "automatic_gefen_fused_binding.cpp",
            "automatic_gefen_fused_kernel.cu",
        ],
    )
    return _EXTENSION_MODULE


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
