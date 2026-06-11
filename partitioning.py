import math

import numpy as np
import torch

DMNB = 4


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
            if i != paired and (max_divisor is None or paired <= max_divisor):
                large.append(paired)
    result = small + large[::-1]
    return result


def _zero_block_mean_error(parameter_name, parameter_shape, period):
    return ZeroBlockMeanError(
        "Encountered a block with mean 0.0, cannot compute element-to-mean ratios. parameter_name={} parameter_shape={} period={}".format(
            parameter_name,
            parameter_shape,
            period,
        )
    )


def average_within_block_variance_cpu(
    values,
    global_mean,
    period_to_check,
    parameter_name=None,
    parameter_shape=None,
    epsilon=1e-12,
):

    del global_mean, parameter_name, parameter_shape

    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if n % period_to_check != 0:
        raise ValueError(
            "Expected length {} to be divisible by period {}.".format(
                n, period_to_check
            )
        )
    elif np.any(values + epsilon <= 0.0):
        raise ValueError(
            "Expected values plus epsilon to be positive before log transform."
        )

    log_values = np.log(values + epsilon)
    blocks = log_values.reshape(-1, period_to_check)
    block_means = blocks.mean(axis=1)
    if len(block_means) < 2:
        raise ValueError(
            "Expected at least two blocks to compare adjacent block means."
        )

    result = np.abs(block_means[1:] - block_means[:-1]).mean()

    return result


def average_within_block_variance_torch(
    values,
    global_mean,
    period,
    parameter_name=None,
    parameter_shape=None,
    epsilon=1e-12,
):
    if not isinstance(values, torch.Tensor):
        raise TypeError("Expected torch.Tensor, got {}".format(type(values).__name__))

    del global_mean, parameter_name, parameter_shape

    values = values.detach().reshape(-1).to(dtype=torch.float64)
    n = values.numel()

    if n % period != 0:
        raise ValueError(
            "Expected length {} to be divisible by period {}.".format(n, period)
        )
    elif torch.any(values + epsilon <= 0.0):
        raise ValueError(
            "Expected values plus epsilon to be positive before log transform."
        )

    log_values = torch.log(values + epsilon)
    blocks = log_values.view(-1, period)
    block_means = blocks.mean(dim=1)
    if block_means.numel() < 2:
        raise ValueError(
            "Expected at least two blocks to compare adjacent block means."
        )

    result = torch.abs(block_means[1:] - block_means[:-1]).mean()

    return result.item()


def _find_period_by_block_variance_cpu(
    squared_grad_flattened_values,
    print_results=True,
    parameter_name=None,
    parameter_shape=None,
    max_period=None,
    threshold=0.25,
    epsilon=1e-12,
    min_num_blocks=DMNB,
    input_is_squared=True,
):
    squared_grad_flattened_values = np.asarray(
        squared_grad_flattened_values, dtype=np.float64
    )
    if not input_is_squared:
        squared_grad_flattened_values = (
            squared_grad_flattened_values * squared_grad_flattened_values
        )
    n = len(squared_grad_flattened_values)

    if np.any(squared_grad_flattened_values + epsilon <= 0.0):
        raise ValueError(
            "Expected values plus epsilon to be positive before log transform."
        )

    global_mean = squared_grad_flattened_values.mean()

    list_period_and_score = []
    divs = divisors(n, max_divisor=max_period)

    for p in divs:
        if n // p < min_num_blocks:
            continue

        score = average_within_block_variance_cpu(
            squared_grad_flattened_values,
            global_mean,
            p,
            parameter_name=parameter_name,
            parameter_shape=parameter_shape,
            epsilon=epsilon,
        )
        list_period_and_score.append((p, score))

    if len(list_period_and_score) == 0:

        return 1

    return _finalize_period_results(list_period_and_score, print_results, n, threshold)


def _find_period_by_block_variance_torch(
    squared_grad_flattened_values,
    print_results=True,
    parameter_name=None,
    parameter_shape=None,
    max_period=None,
    threshold=0.25,
    epsilon=1e-12,
    min_num_blocks=DMNB,
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

    values = squared_grad_flattened_values.detach().reshape(-1).to(dtype=torch.float64)
    n = values.numel()

    if torch.any(values < 0.0):
        raise ValueError("Expected non-negative values before log transform.")

    global_mean = values.mean()

    results = []
    divs = divisors(n, max_divisor=max_period)

    for p in divs:
        if n // p < min_num_blocks:
            continue
        else:
            score = average_within_block_variance_torch(
                values,
                global_mean,
                p,
                parameter_name=parameter_name,
                parameter_shape=parameter_shape,
                epsilon=epsilon,
            )
        results.append((p, score))

    if len(results) == 0:

        return 1

    return _finalize_period_results(results, print_results, n, threshold)


def _find_period_by_block_variance_cuda_kernel(
    values,
    print_results=True,
    parameter_name=None,
    parameter_shape=None,
    input_is_squared=True,
    max_period=None,
    threshold=0.25,
    epsilon=1e-12,
    min_num_blocks=DMNB,
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
        average_adjacent_log_block_mean_difference_cuda_kernel,
    )

    for p in divs:
        if n // p < min_num_blocks:
            continue
        score = average_adjacent_log_block_mean_difference_cuda_kernel(
            flattened_values,
            p,
            input_is_squared=input_is_squared,
            epsilon=epsilon,
        )
        results.append((p, score))

    if len(results) == 0:

        return 1

    return _finalize_period_results(results, print_results, n, threshold)


def _finalize_period_results(
    list_periods_and_scores, print_results, flattened_length, threshold
):

    if len(list_periods_and_scores) == 0:
        raise ValueError("Expected at least one valid period candidate.")

    scored_results = []
    for index, (p, score) in enumerate(list_periods_and_scores):
        if index == len(list_periods_and_scores) - 1:
            score_gap = 0.0
        elif index < len(list_periods_and_scores) - 1:
            next_score = list_periods_and_scores[index + 1][1]
            if next_score == 0.0:
                if score == 0.0:
                    score_gap = 0.0
                elif score != 0.0:
                    score_gap = float("inf")
                else:
                    raise ValueError("Unexpected zero score gap state.")
            else:
                score_gap = score / next_score
        else:
            raise ValueError("Unexpected score gap state.")
        scored_results.append((p, score, score_gap))

    del threshold

    last_period = list_periods_and_scores[-1][0]

    score_candidates = []
    for index in range(1, len(list_periods_and_scores) - 1):
        p, score = list_periods_and_scores[index]
        previous_score = list_periods_and_scores[index - 1][1]
        next_score = list_periods_and_scores[index + 1][1]
        if p <= 8:
            continue
        elif p == last_period:
            continue
        elif score <= previous_score:
            continue
        elif score <= next_score:
            continue
        score_candidates.append((score, p))
    if len(score_candidates) == 0:
        best_period = 1
    else:
        best_score, best_period = max(score_candidates)
        if not math.isfinite(best_score):
            raise ValueError("Expected finite best score, got {}.".format(best_score))

    PC = 1024
    if best_period > PC:

        best_period = min(
            p
            for p in divisors(flattened_length)
            if p > PC and p <= best_period and best_period % p == 0
        )

    if print_results:
        print(f"Flattened length: {flattened_length}")
        for p, score, score_gap in scored_results:

            line = f"p={p} |  score={score:.10f} | score gap = {score_gap:.3f}"
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
    threshold=0.25,
    epsilon=1e-12,
    min_num_blocks=DMNB,
):

    if backend == "cpu":
        return _find_period_by_block_variance_cpu(
            squared_grad_flattened_values,
            print_results=print_results,
            parameter_name=parameter_name,
            parameter_shape=parameter_shape,
            max_period=max_period,
            threshold=threshold,
            epsilon=epsilon,
            min_num_blocks=min_num_blocks,
            input_is_squared=input_is_squared,
        )
    elif backend == "gpu":
        return _find_period_by_block_variance_torch(
            squared_grad_flattened_values,
            print_results=print_results,
            parameter_name=parameter_name,
            parameter_shape=parameter_shape,
            max_period=max_period,
            threshold=threshold,
            epsilon=epsilon,
            min_num_blocks=min_num_blocks,
        )
    elif backend == "cuda_kernel":
        return _find_period_by_block_variance_cuda_kernel(
            squared_grad_flattened_values,
            print_results=print_results,
            parameter_name=parameter_name,
            parameter_shape=parameter_shape,
            input_is_squared=input_is_squared,
            max_period=max_period,
            threshold=threshold,
            epsilon=epsilon,
            min_num_blocks=min_num_blocks,
        )
    raise ValueError("Unexpected backend: {}".format(backend))
