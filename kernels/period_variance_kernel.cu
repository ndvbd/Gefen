#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMacros.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <cmath>
#include <stdexcept>

namespace {

template <typename scalar_t, bool input_is_squared>
__global__ void average_within_block_variance_kernel(
    const scalar_t* __restrict__ values,
    int64_t period,
    double* __restrict__ out_sum_var
) {
    extern __shared__ float shared[];
    float* shared_sum_x = shared;
    float* shared_sum_x2 = shared + blockDim.x;

    const int64_t logical_block = static_cast<int64_t>(blockIdx.x);
    const int64_t start = logical_block * period;

    float local_sum_x = 0.0f;
    float local_sum_x2 = 0.0f;
    for (int64_t offset = threadIdx.x; offset < period; offset += blockDim.x) {
        const float value = static_cast<float>(values[start + offset]);
        const float x = input_is_squared ? value : value * value;
        local_sum_x += x;
        local_sum_x2 += x * x;
    }

    shared_sum_x[threadIdx.x] = local_sum_x;
    shared_sum_x2[threadIdx.x] = local_sum_x2;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum_x[threadIdx.x] += shared_sum_x[threadIdx.x + stride];
            shared_sum_x2[threadIdx.x] += shared_sum_x2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float mean_x = shared_sum_x[0] / static_cast<float>(period);
        float var_x = shared_sum_x2[0] / static_cast<float>(period);
        var_x -= mean_x * mean_x;
        if (var_x < 0.0f) {
            var_x = 0.0f;
        }
        atomicAdd(out_sum_var, static_cast<double>(var_x));
    }
}

template <typename scalar_t, bool input_is_squared>
__global__ void average_within_block_coefficient_of_variation_kernel(
    const scalar_t* __restrict__ values,
    int64_t period,
    double* __restrict__ out_stats
) {
    extern __shared__ float shared[];
    float* shared_sum_x = shared;
    float* shared_sum_x2 = shared + blockDim.x;

    const int64_t logical_block = static_cast<int64_t>(blockIdx.x);
    const int64_t start = logical_block * period;

    float local_sum_x = 0.0f;
    float local_sum_x2 = 0.0f;
    for (int64_t offset = threadIdx.x; offset < period; offset += blockDim.x) {
        const float value = static_cast<float>(values[start + offset]);
        const float x = input_is_squared ? value : value * value;
        local_sum_x += x;
        local_sum_x2 += x * x;
    }

    shared_sum_x[threadIdx.x] = local_sum_x;
    shared_sum_x2[threadIdx.x] = local_sum_x2;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum_x[threadIdx.x] += shared_sum_x[threadIdx.x + stride];
            shared_sum_x2[threadIdx.x] += shared_sum_x2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float mean_x = shared_sum_x[0] / static_cast<float>(period);
        if (mean_x < 0.0f) {
            atomicAdd(out_stats + 2, 1.0);
        } else if (mean_x > 0.0f) {
            float var_x = shared_sum_x2[0] / static_cast<float>(period);
            var_x -= mean_x * mean_x;
            if (var_x < 0.0f) {
                var_x = 0.0f;
            }
            atomicAdd(out_stats, static_cast<double>(sqrtf(var_x) / mean_x));
            atomicAdd(out_stats + 1, 1.0);
        }
    }
}

template <typename scalar_t, bool input_is_squared>
__global__ void average_adjacent_log_block_mean_difference_kernel(
    const scalar_t* __restrict__ values,
    int64_t period,
    double epsilon,
    double* __restrict__ out_stats
) {
    extern __shared__ unsigned char adjacent_shared_raw[];
    double* shared_sum_current = reinterpret_cast<double*>(adjacent_shared_raw);
    double* shared_sum_next = shared_sum_current + blockDim.x;
    double* shared_invalid_count = shared_sum_current + 2 * blockDim.x;

    const int64_t adjacent_block = static_cast<int64_t>(blockIdx.x);
    const int64_t current_start = adjacent_block * period;
    const int64_t next_start = current_start + period;

    double local_sum_current = 0.0;
    double local_sum_next = 0.0;
    double local_invalid_count = 0.0;
    for (int64_t offset = threadIdx.x; offset < period; offset += blockDim.x) {
        const double current_value_raw = static_cast<double>(static_cast<float>(values[current_start + offset]));
        const double next_value_raw = static_cast<double>(static_cast<float>(values[next_start + offset]));
        const double current_value = input_is_squared ? current_value_raw : current_value_raw * current_value_raw;
        const double next_value = input_is_squared ? next_value_raw : next_value_raw * next_value_raw;
        const double current_log_input = current_value + epsilon;
        const double next_log_input = next_value + epsilon;
        if (current_log_input <= 0.0 || next_log_input <= 0.0) {
            local_invalid_count += 1.0;
        } else {
            local_sum_current += log(current_log_input);
            local_sum_next += log(next_log_input);
        }
    }

    shared_sum_current[threadIdx.x] = local_sum_current;
    shared_sum_next[threadIdx.x] = local_sum_next;
    shared_invalid_count[threadIdx.x] = local_invalid_count;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum_current[threadIdx.x] += shared_sum_current[threadIdx.x + stride];
            shared_sum_next[threadIdx.x] += shared_sum_next[threadIdx.x + stride];
            shared_invalid_count[threadIdx.x] += shared_invalid_count[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (shared_invalid_count[0] > 0.0) {
            atomicAdd(out_stats + 1, shared_invalid_count[0]);
        } else {
            const double current_mean = shared_sum_current[0] / static_cast<double>(period);
            const double next_mean = shared_sum_next[0] / static_cast<double>(period);
            atomicAdd(out_stats, fabs(next_mean - current_mean));
        }
    }
}

int next_power_of_two(int value) {
    int power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
}

}  // namespace

at::Tensor average_within_block_variance_cuda(
    at::Tensor values,
    int64_t period,
    bool input_is_squared
) {
    if (!values.is_cuda()) {
        throw std::invalid_argument("Expected a CUDA tensor.");
    }
    if (!values.is_contiguous()) {
        throw std::invalid_argument("Expected a contiguous tensor.");
    }
    if (!values.is_floating_point()) {
        throw std::invalid_argument("Expected a floating-point tensor.");
    }
    if (values.dim() != 1) {
        throw std::invalid_argument("Expected a flat 1D tensor.");
    }
    if (period <= 0) {
        throw std::invalid_argument("Expected a positive period.");
    }
    if (values.numel() % period != 0) {
        throw std::invalid_argument("Expected tensor length to be divisible by period.");
    }

    c10::cuda::CUDAGuard device_guard(values.device());

    const int64_t logical_blocks = values.numel() / period;
    auto result = at::zeros({1}, values.options().dtype(at::kDouble));
    if (logical_blocks == 0) {
        return result;
    }

    int threads = next_power_of_two(static_cast<int>(period));
    if (threads > 256) {
        threads = 256;
    } else if (threads < 32) {
        threads = 32;
    }
    const dim3 grid(static_cast<unsigned int>(logical_blocks));
    const dim3 block(static_cast<unsigned int>(threads));
    const size_t shared_bytes = static_cast<size_t>(threads) * 2 * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        values.scalar_type(),
        "average_within_block_variance_cuda",
        [&] {
            if (input_is_squared) {
                average_within_block_variance_kernel<scalar_t, true><<<grid, block, shared_bytes>>>(
                    values.data_ptr<scalar_t>(),
                    period,
                    result.data_ptr<double>()
                );
            } else {
                average_within_block_variance_kernel<scalar_t, false><<<grid, block, shared_bytes>>>(
                    values.data_ptr<scalar_t>(),
                    period,
                    result.data_ptr<double>()
                );
            }
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    result.div_(static_cast<double>(logical_blocks)).sqrt_();
    return result;
}

at::Tensor average_within_block_coefficient_of_variation_cuda(
    at::Tensor values,
    int64_t period,
    bool input_is_squared
) {
    if (!values.is_cuda()) {
        throw std::invalid_argument("Expected a CUDA tensor.");
    }
    if (!values.is_contiguous()) {
        throw std::invalid_argument("Expected a contiguous tensor.");
    }
    if (!values.is_floating_point()) {
        throw std::invalid_argument("Expected a floating-point tensor.");
    }
    if (values.dim() != 1) {
        throw std::invalid_argument("Expected a flat 1D tensor.");
    }
    if (period <= 0) {
        throw std::invalid_argument("Expected a positive period.");
    }
    if (values.numel() % period != 0) {
        throw std::invalid_argument("Expected tensor length to be divisible by period.");
    }

    c10::cuda::CUDAGuard device_guard(values.device());

    const int64_t logical_blocks = values.numel() / period;
    auto stats = at::zeros({3}, values.options().dtype(at::kDouble));
    if (logical_blocks == 0) {
        return stats;
    }

    int threads = next_power_of_two(static_cast<int>(period));
    if (threads > 256) {
        threads = 256;
    } else if (threads < 32) {
        threads = 32;
    }
    const dim3 grid(static_cast<unsigned int>(logical_blocks));
    const dim3 block(static_cast<unsigned int>(threads));
    const size_t shared_bytes = static_cast<size_t>(threads) * 2 * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        values.scalar_type(),
        "average_within_block_coefficient_of_variation_cuda",
        [&] {
            if (input_is_squared) {
                average_within_block_coefficient_of_variation_kernel<scalar_t, true><<<grid, block, shared_bytes>>>(
                    values.data_ptr<scalar_t>(),
                    period,
                    stats.data_ptr<double>()
                );
            } else {
                average_within_block_coefficient_of_variation_kernel<scalar_t, false><<<grid, block, shared_bytes>>>(
                    values.data_ptr<scalar_t>(),
                    period,
                    stats.data_ptr<double>()
                );
            }
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (stats[2].item<double>() > 0.0) {
        throw std::runtime_error("Expected non-negative block means for coefficient of variation.");
    }
    if (stats[1].item<double>() == 0.0) {
        throw std::runtime_error("Encountered only zero block means, cannot compute coefficient of variation.");
    }
    return stats[0].div(stats[1]);
}

at::Tensor average_adjacent_log_block_mean_difference_cuda(
    at::Tensor values,
    int64_t period,
    bool input_is_squared,
    double epsilon
) {
    if (!values.is_cuda()) {
        throw std::invalid_argument("Expected a CUDA tensor.");
    }
    if (!values.is_contiguous()) {
        throw std::invalid_argument("Expected a contiguous tensor.");
    }
    if (!values.is_floating_point()) {
        throw std::invalid_argument("Expected a floating-point tensor.");
    }
    if (values.dim() != 1) {
        throw std::invalid_argument("Expected a flat 1D tensor.");
    }
    if (period <= 0) {
        throw std::invalid_argument("Expected a positive period.");
    }
    if (epsilon <= 0.0) {
        throw std::invalid_argument("Expected a positive epsilon.");
    }
    if (values.numel() % period != 0) {
        throw std::invalid_argument("Expected tensor length to be divisible by period.");
    }

    c10::cuda::CUDAGuard device_guard(values.device());

    const int64_t logical_blocks = values.numel() / period;
    if (logical_blocks < 2) {
        throw std::invalid_argument("Expected at least two blocks to compare adjacent block means.");
    }

    auto stats = at::zeros({2}, values.options().dtype(at::kDouble));

    int threads = next_power_of_two(static_cast<int>(period));
    if (threads > 256) {
        threads = 256;
    } else if (threads < 32) {
        threads = 32;
    }
    const dim3 grid(static_cast<unsigned int>(logical_blocks - 1));
    const dim3 block(static_cast<unsigned int>(threads));
    const size_t shared_bytes = static_cast<size_t>(threads) * 3 * sizeof(double);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        values.scalar_type(),
        "average_adjacent_log_block_mean_difference_cuda",
        [&] {
            if (input_is_squared) {
                average_adjacent_log_block_mean_difference_kernel<scalar_t, true><<<grid, block, shared_bytes>>>(
                    values.data_ptr<scalar_t>(),
                    period,
                    epsilon,
                    stats.data_ptr<double>()
                );
            } else {
                average_adjacent_log_block_mean_difference_kernel<scalar_t, false><<<grid, block, shared_bytes>>>(
                    values.data_ptr<scalar_t>(),
                    period,
                    epsilon,
                    stats.data_ptr<double>()
                );
            }
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (stats[1].item<double>() > 0.0) {
        throw std::runtime_error("Expected values plus epsilon to be positive before log transform.");
    }
    return stats[0].div(static_cast<double>(logical_blocks - 1));
}
