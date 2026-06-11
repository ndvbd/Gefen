import torch

from .gefen_cuda import load_extension

_OPAQUE_OP_AVAILABLE = False


def _load_extension():
    return load_extension()


def _average_within_block_variance_cuda_impl(
    values: torch.Tensor,
    period: int,
    input_is_squared: bool,
) -> torch.Tensor:
    module = _load_extension()
    flattened = values.detach().reshape(-1).contiguous()
    return module.average_within_block_variance_cuda(
        flattened, period, input_is_squared
    )


def _average_within_block_coefficient_of_variation_cuda_impl(
    values: torch.Tensor,
    period: int,
    input_is_squared: bool,
) -> torch.Tensor:
    module = _load_extension()
    flattened = values.detach().reshape(-1).contiguous()
    return module.average_within_block_coefficient_of_variation_cuda(
        flattened, period, input_is_squared
    )


if hasattr(torch.library, "custom_op"):

    @torch.library.custom_op(
        "gefen::average_within_block_variance_cuda", mutates_args=()
    )
    def _average_within_block_variance_cuda_opaque(
        values: torch.Tensor,
        period: int,
        input_is_squared: bool,
    ) -> torch.Tensor:
        return _average_within_block_variance_cuda_impl(
            values, period, input_is_squared
        )

    @_average_within_block_variance_cuda_opaque.register_fake
    def _average_within_block_variance_cuda_opaque_fake(
        values: torch.Tensor,
        period: int,
        input_is_squared: bool,
    ) -> torch.Tensor:
        del period, input_is_squared
        return values.new_empty(())

    _OPAQUE_OP_AVAILABLE = True


def average_within_block_variance_cuda_kernel(
    values: torch.Tensor,
    period: int,
    *,
    input_is_squared: bool,
) -> float:
    if not isinstance(values, torch.Tensor):
        raise TypeError("Expected torch.Tensor, got {}".format(type(values).__name__))
    elif values.device.type != "cuda":
        raise ValueError("Expected a CUDA tensor, got device {}".format(values.device))
    elif values.numel() == 0:
        raise ValueError("Expected a non-empty tensor.")
    elif period <= 0:
        raise ValueError("Expected a positive period, got {}".format(period))
    elif values.numel() % period != 0:
        raise ValueError(
            "Expected length {} to be divisible by period {}.".format(
                values.numel(), period
            )
        )

    result = _average_within_block_variance_cuda_impl(values, period, input_is_squared)
    return float(result.item())


def average_within_block_coefficient_of_variation_cuda_kernel(
    values: torch.Tensor,
    period: int,
    *,
    input_is_squared: bool,
) -> float:
    if not isinstance(values, torch.Tensor):
        raise TypeError("Expected torch.Tensor, got {}".format(type(values).__name__))
    elif values.device.type != "cuda":
        raise ValueError("Expected a CUDA tensor, got device {}".format(values.device))
    elif values.numel() == 0:
        raise ValueError("Expected a non-empty tensor.")
    elif period <= 0:
        raise ValueError("Expected a positive period, got {}".format(period))
    elif values.numel() % period != 0:
        raise ValueError(
            "Expected length {} to be divisible by period {}.".format(
                values.numel(), period
            )
        )

    result = _average_within_block_coefficient_of_variation_cuda_impl(
        values, period, input_is_squared
    )
    return float(result.item())
