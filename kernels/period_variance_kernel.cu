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
    float* __restrict__ out_sum_var
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
        atomicAdd(out_sum_var, var_x);
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
    auto result = at::zeros({1}, values.options().dtype(at::kFloat));
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
                    result.data_ptr<float>()
                );
            } else {
                average_within_block_variance_kernel<scalar_t, false><<<grid, block, shared_bytes>>>(
                    values.data_ptr<scalar_t>(),
                    period,
                    result.data_ptr<float>()
                );
            }
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    result.div_(static_cast<double>(logical_blocks)).sqrt_();
    return result;
}
