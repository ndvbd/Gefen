import torch

from ._build_notice import load_gefen_cuda_extension

_EXTENSION_MODULE = None


def _load_extension():
    global _EXTENSION_MODULE
    if _EXTENSION_MODULE is not None:
        return _EXTENSION_MODULE

    _EXTENSION_MODULE = load_gefen_cuda_extension(
        "gefen_automatic_vmean_ext",
        [
            "automatic_vmean_binding.cpp",
            "automatic_vmean_kernel.cu",
        ],
    )
    return _EXTENSION_MODULE


def automatic_vmean_update_cuda(
    vmean: torch.Tensor,
    grad_view: torch.Tensor,
    beta2: float,
) -> None:
    if not isinstance(vmean, torch.Tensor):
        raise TypeError("Expected vmean to be a torch.Tensor.")
    elif not isinstance(grad_view, torch.Tensor):
        raise TypeError("Expected grad_view to be a torch.Tensor.")
    elif vmean.device.type != "cuda":
        raise ValueError(
            "Expected vmean to be on CUDA, got device {}.".format(vmean.device)
        )
    elif grad_view.device.type != "cuda":
        raise ValueError(
            "Expected grad_view to be on CUDA, got device {}.".format(grad_view.device)
        )
    elif grad_view.dim() != 2:
        raise ValueError(
            "Expected grad_view to be 2D, got dim={}.".format(grad_view.dim())
        )
    elif vmean.dim() != 2 or vmean.shape[1] != 1:
        raise ValueError(
            "Expected vmean to have shape [num_blocks, 1], got {}.".format(
                tuple(vmean.shape)
            )
        )
    elif grad_view.shape[0] != vmean.shape[0]:
        raise ValueError(
            "Expected grad_view and vmean to have the same number of blocks."
        )
    elif vmean.dtype != torch.float32:
        raise ValueError("Expected vmean to be float32, got {}.".format(vmean.dtype))
    elif not 0.0 <= beta2 <= 1.0:
        raise ValueError("Expected beta2 to be in [0, 1], got {}.".format(beta2))

    module = _load_extension()
    module.automatic_vmean_update_cuda(
        vmean,
        grad_view.contiguous(),
        beta2,
    )
