import math

import numpy as np
import torch


class ZeroBlockMeanError(ValueError):
    pass


def divisors(n, max_divisor=None):

    if max_divisor is not None:
        max_divisor = int(max_divisor)
        if max_divisor < 1:
            raise ValueError(
                "max_divisor must be positive, got {}.".format(max_divisor)
            )
    small, large = [], []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            if i != n and (max_divisor is None or i <= max_divisor):
                small.append(i)
            paired = n // i
            if (
                i != paired
                and paired != n
                and (max_divisor is None or paired <= max_divisor)
            ):
                large.append(paired)
    return small + large[::-1]


def _zero_block_mean_error(parameter_name, parameter_shape, period):
    return ZeroBlockMeanError(
        "Encountered a block with mean 0.0, cannot compute element-to-mean ratios. parameter_name={} parameter_shape={} period={}".format(
            parameter_name,
            parameter_shape,
            period,
        )
    )


def average_within_block_variance_cpu(
    list_of_values, period_to_check, parameter_name=None, parameter_shape=None
):

    list_of_values = np.asarray(list_of_values, dtype=float)
    n = len(list_of_values)

    if n % period_to_check != 0:
        raise ValueError(
            "Expected length {} to be divisible by period {}.".format(
                n, period_to_check
            )
        )

    blocks = list_of_values.reshape(-1, period_to_check)
    block_means = blocks.mean(axis=1)
    if np.any(block_means < 0.0):
        raise ValueError(
            "Expected non-negative block means for coefficient of variation."
        )

    non_zero_mean_mask = block_means != 0.0
    if not np.any(non_zero_mean_mask):
        raise _zero_block_mean_error(parameter_name, parameter_shape, period_to_check)

    valid_blocks = blocks[non_zero_mean_mask]
    valid_block_means = block_means[non_zero_mean_mask]

    block_coefficients = valid_blocks.std(axis=1) / valid_block_means
    result = block_coefficients.mean()

    return result


def average_within_block_variance_torch(
    values, period, parameter_name=None, parameter_shape=None
):
    if not isinstance(values, torch.Tensor):
        raise TypeError("Expected torch.Tensor, got {}".format(type(values).__name__))

    values = values.detach().reshape(-1).to(dtype=torch.float32)
    n = values.numel()

    if n % period != 0:
        raise ValueError(
            "Expected length {} to be divisible by period {}.".format(n, period)
        )

    blocks = values.view(-1, period)
    block_means = blocks.mean(dim=1)
    if torch.any(block_means < 0.0):
        raise ValueError(
            "Expected non-negative block means for coefficient of variation."
        )

    non_zero_mean_mask = block_means != 0.0
    if not torch.any(non_zero_mean_mask):
        raise _zero_block_mean_error(parameter_name, parameter_shape, period)

    valid_blocks = blocks[non_zero_mean_mask]
    valid_block_means = block_means[non_zero_mean_mask]

    block_coefficients = valid_blocks.std(dim=1, correction=0) / valid_block_means
    result = block_coefficients.mean()

    return result.item()


def _find_period_by_block_variance_cpu(
    squared_grad_flattened_values,
    print_results=True,
    parameter_name=None,
    parameter_shape=None,
    max_period=None,
):
    squared_grad_flattened_values = np.asarray(
        squared_grad_flattened_values, dtype=float
    )
    n = len(squared_grad_flattened_values)

    list_period_and_error = []
    divs = divisors(n, max_divisor=max_period)

    for p in divs:
        try:
            err_average_variance_in_blocks = average_within_block_variance_cpu(
                squared_grad_flattened_values,
                p,
                parameter_name=parameter_name,
                parameter_shape=parameter_shape,
            )
        except ZeroBlockMeanError:
            continue
        list_period_and_error.append((p, err_average_variance_in_blocks))

    if len(list_period_and_error) == 0:
        raise ValueError(
            "Could not find any valid period candidates. parameter_name={} parameter_shape={}".format(
                parameter_name,
                parameter_shape,
            )
        )

    return _finalize_period_results(list_period_and_error, print_results)


def _find_period_by_block_variance_torch(
    squared_grad_flattened_values,
    print_results=True,
    parameter_name=None,
    parameter_shape=None,
    max_period=None,
):
    if not isinstance(squared_grad_flattened_values, torch.Tensor):
        raise TypeError(
            "Expected torch.Tensor for gpu backend, got {}".format(
                type(squared_grad_flattened_values).__name__
            )
        )
    if squared_grad_flattened_values.device.type != "cuda":
        raise ValueError(
            "Expected a CUDA tensor for gpu backend, got device {}".format(
                squared_grad_flattened_values.device
            )
        )

    values = squared_grad_flattened_values.detach().reshape(-1).to(dtype=torch.float32)
    n = values.numel()

    results = []
    divs = divisors(n, max_divisor=max_period)

    for p in divs:
        try:
            err = average_within_block_variance_torch(
                values,
                p,
                parameter_name=parameter_name,
                parameter_shape=parameter_shape,
            )
        except ZeroBlockMeanError:

            continue
        results.append((p, err))

    if len(results) == 0:
        raise ValueError(
            "Could not find any valid period candidates. parameter_name={} parameter_shape={}".format(
                parameter_name,
                parameter_shape,
            )
        )

    return _finalize_period_results(results, print_results)


def _find_period_by_block_variance_cuda_kernel(
    values,
    print_results=True,
    parameter_name=None,
    parameter_shape=None,
    input_is_squared=True,
    max_period=None,
):
    if not isinstance(values, torch.Tensor):
        raise TypeError(
            "Expected torch.Tensor for cuda_kernel backend, got {}".format(
                type(values).__name__
            )
        )
    if values.device.type != "cuda":
        raise ValueError(
            "Expected a CUDA tensor for cuda_kernel backend, got device {}".format(
                values.device
            )
        )

    flattened_values = values.detach().reshape(-1)
    n = flattened_values.numel()

    results = []
    divs = divisors(n, max_divisor=max_period)

    from gefen.kernels.period_variance import (
        average_within_block_coefficient_of_variation_cuda_kernel,
    )

    for p in divs:
        err = average_within_block_coefficient_of_variation_cuda_kernel(
            flattened_values,
            p,
            input_is_squared=input_is_squared,
        )

        results.append((p, err))

    if len(results) == 0:
        raise ValueError(
            "Could not find any valid period candidates. parameter_name={} parameter_shape={}".format(
                parameter_name,
                parameter_shape,
            )
        )

    return _finalize_period_results(results, print_results)


def _finalize_period_results(list_periods_and_errors, print_results):

    if len(list_periods_and_errors) == 0:
        raise ValueError("Expected at least one valid period candidate.")

    scored_results = []
    previous_err = None
    eps = 1e-3

    for p, err in list_periods_and_errors:
        if previous_err is None:
            d_error = eps
        else:
            if previous_err == 0.0:
                d_error = float("inf")
            else:
                d_error = err / previous_err
        previous_err = err

        scored_results.append((p, err, d_error))

    period_candidates = scored_results[1:]
    if len(period_candidates) == 0:
        best_period = 1
    else:
        min_d_error_without_first, min_d_error_period, min_d_error_actual_error = min(
            (_d_error, p, err) for p, err, _d_error in period_candidates
        )
        if min_d_error_without_first < 1.10 and min_d_error_actual_error < 3.0:
            best_period = min_d_error_period
        else:
            best_period = 1

    if best_period < 8:
        best_period = 1

    if print_results:
        for p, err, d_error in scored_results:
            line = f"p={p} |  err={err:.15f} | d error = {d_error:.15f}"
            if p == best_period:
                print(f"\033[92m{line}\033[0m")
            elif p != best_period:
                print(line)

    return best_period


def find_period_by_block_variance(
    squared_grad_flattened_values,
    print_results=True,
    parameter_name=None,
    parameter_shape=None,
    backend="cpu",
    input_is_squared=True,
    max_period=None,
):
    if backend == "cpu":
        return _find_period_by_block_variance_cpu(
            squared_grad_flattened_values,
            print_results=print_results,
            parameter_name=parameter_name,
            parameter_shape=parameter_shape,
            max_period=max_period,
        )
    elif backend == "gpu":
        return _find_period_by_block_variance_torch(
            squared_grad_flattened_values,
            print_results=print_results,
            parameter_name=parameter_name,
            parameter_shape=parameter_shape,
            max_period=max_period,
        )
    elif backend == "cuda_kernel":
        return _find_period_by_block_variance_cuda_kernel(
            squared_grad_flattened_values,
            print_results=print_results,
            parameter_name=parameter_name,
            parameter_shape=parameter_shape,
            input_is_squared=input_is_squared,
            max_period=max_period,
        )
    raise ValueError("Unexpected backend: {}".format(backend))
