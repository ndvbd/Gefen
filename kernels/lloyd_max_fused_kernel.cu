#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMacros.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr int kMaxThreads = 256;
constexpr int kMaxCodebookSize = 256;

__device__ __forceinline__ uint8_t nearest_codebook_index(
    float normalized_value,
    const float* __restrict__ codebook,
    int codebook_size
) {
    int left = 0;
    int right = codebook_size;
    while (left < right) {
        const int mid = left + (right - left) / 2;
        if (codebook[mid] < normalized_value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    int right_idx = left;
    if (right_idx < 0) {
        right_idx = 0;
    } else if (right_idx >= codebook_size) {
        right_idx = codebook_size - 1;
    }

    int left_idx = right_idx - 1;
    if (left_idx < 0) {
        left_idx = 0;
    }

    const float left_dist = fabsf(normalized_value - codebook[left_idx]);
    const float right_dist = fabsf(normalized_value - codebook[right_idx]);
    return static_cast<uint8_t>(left_dist <= right_dist ? left_idx : right_idx);
}

template <typename scalar_t>
__global__ void gefen_lloyd_accumulate_kernel(
    const scalar_t* __restrict__ grad_flat,
    const float* __restrict__ codebook,
    int codebook_size,
    int64_t period,
    int64_t num_blocks,
    float* __restrict__ sums,
    int64_t* __restrict__ counts
) {
    __shared__ float shared_absmax[kMaxThreads];
    __shared__ float shared_sums[kMaxCodebookSize];
    __shared__ int shared_counts[kMaxCodebookSize];

    const int64_t logical_block_idx = static_cast<int64_t>(blockIdx.x);
    if (logical_block_idx >= num_blocks) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int64_t start = logical_block_idx * period;

    for (int idx = tid; idx < codebook_size; idx += blockDim.x) {
        shared_sums[idx] = 0.0f;
        shared_counts[idx] = 0;
    }

    float local_absmax = 0.0f;
    for (int64_t offset = tid; offset < period; offset += blockDim.x) {
        const float value = static_cast<float>(grad_flat[start + offset]);
        const float abs_value = fabsf(value);
        if (abs_value > local_absmax) {
            local_absmax = abs_value;
        }
    }

    shared_absmax[tid] = local_absmax;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < static_cast<int>(stride) && shared_absmax[tid + stride] > shared_absmax[tid]) {
            shared_absmax[tid] = shared_absmax[tid + stride];
        }
        __syncthreads();
    }

    const float absmax = shared_absmax[0];
    __syncthreads();

    for (int64_t offset = tid; offset < period; offset += blockDim.x) {
        const float grad_value = static_cast<float>(grad_flat[start + offset]);
        float normalized_value = 0.0f;
        if (absmax > 0.0f) {
            normalized_value = grad_value / absmax;
        }
        const uint8_t assignment = nearest_codebook_index(normalized_value, codebook, codebook_size);
        atomicAdd(&shared_sums[assignment], normalized_value);
        atomicAdd(&shared_counts[assignment], 1);
    }
    __syncthreads();

    for (int idx = tid; idx < codebook_size; idx += blockDim.x) {
        atomicAdd(&sums[idx], shared_sums[idx]);
        atomicAdd(
            reinterpret_cast<unsigned long long*>(&counts[idx]),
            static_cast<unsigned long long>(shared_counts[idx])
        );
    }
}

template <typename scalar_t>
__global__ void gefen_lloyd_mse_kernel(
    const scalar_t* __restrict__ grad_flat,
    const float* __restrict__ old_codebook,
    const float* __restrict__ new_codebook,
    int codebook_size,
    int64_t period,
    int64_t num_blocks,
    float* __restrict__ mse_normalized_sum,
    float* __restrict__ mse_original_sum
) {
    __shared__ float shared_absmax[kMaxThreads];
    __shared__ float shared_norm_error[kMaxThreads];
    __shared__ float shared_orig_error[kMaxThreads];

    const int64_t logical_block_idx = static_cast<int64_t>(blockIdx.x);
    if (logical_block_idx >= num_blocks) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int64_t start = logical_block_idx * period;

    float local_absmax = 0.0f;
    for (int64_t offset = tid; offset < period; offset += blockDim.x) {
        const float value = static_cast<float>(grad_flat[start + offset]);
        const float abs_value = fabsf(value);
        if (abs_value > local_absmax) {
            local_absmax = abs_value;
        }
    }

    shared_absmax[tid] = local_absmax;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < static_cast<int>(stride) && shared_absmax[tid + stride] > shared_absmax[tid]) {
            shared_absmax[tid] = shared_absmax[tid + stride];
        }
        __syncthreads();
    }

    const float absmax = shared_absmax[0];

    float local_norm_error = 0.0f;
    float local_orig_error = 0.0f;
    for (int64_t offset = tid; offset < period; offset += blockDim.x) {
        const float grad_value = static_cast<float>(grad_flat[start + offset]);
        float normalized_value = 0.0f;
        if (absmax > 0.0f) {
            normalized_value = grad_value / absmax;
        }
        const uint8_t assignment = nearest_codebook_index(normalized_value, old_codebook, codebook_size);
        const float error = normalized_value - new_codebook[assignment];
        local_norm_error += error * error;
        const float original_error = error * absmax;
        local_orig_error += original_error * original_error;
    }

    shared_norm_error[tid] = local_norm_error;
    shared_orig_error[tid] = local_orig_error;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < static_cast<int>(stride)) {
            shared_norm_error[tid] += shared_norm_error[tid + stride];
            shared_orig_error[tid] += shared_orig_error[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(mse_normalized_sum, shared_norm_error[0]);
        atomicAdd(mse_original_sum, shared_orig_error[0]);
    }
}

int choose_threads(int64_t period) {
    int threads = 32;
    while (threads < period && threads < kMaxThreads) {
        threads <<= 1;
    }
    if (threads > kMaxThreads) {
        threads = kMaxThreads;
    }
    return threads;
}

void validate_common_inputs(
    const at::Tensor& grad_flat,
    const at::Tensor& codebook,
    int64_t period
) {
    if (!grad_flat.is_cuda() || !codebook.is_cuda()) {
        throw std::invalid_argument("Expected grad_flat and codebook to be CUDA tensors.");
    }
    if (!grad_flat.is_contiguous()) {
        throw std::invalid_argument("Expected grad_flat to be contiguous.");
    }
    if (!codebook.is_contiguous()) {
        throw std::invalid_argument("Expected codebook to be contiguous.");
    }
    if (grad_flat.dim() != 1) {
        throw std::invalid_argument("Expected grad_flat to be 1D.");
    }
    if (codebook.dim() != 1) {
        throw std::invalid_argument("Expected codebook to be 1D.");
    }
    if (codebook.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected codebook to have dtype float32.");
    }
    if (codebook.numel() <= 0 || codebook.numel() > kMaxCodebookSize) {
        throw std::invalid_argument("Expected codebook size in [1, 256].");
    }
    if (period <= 0) {
        throw std::invalid_argument("Expected period to be positive.");
    }
    if (grad_flat.numel() % period != 0) {
        throw std::invalid_argument("Expected grad_flat.numel() to be divisible by period.");
    }
}

}  // namespace

void gefen_lloyd_accumulate_cuda(
    at::Tensor grad_flat,
    at::Tensor codebook,
    int64_t period,
    at::Tensor sums,
    at::Tensor counts
) {
    validate_common_inputs(grad_flat, codebook, period);
    if (!sums.is_cuda() || !counts.is_cuda()) {
        throw std::invalid_argument("Expected sums and counts to be CUDA tensors.");
    }
    if (!sums.is_contiguous() || !counts.is_contiguous()) {
        throw std::invalid_argument("Expected sums and counts to be contiguous.");
    }
    if (sums.dim() != 1 || counts.dim() != 1) {
        throw std::invalid_argument("Expected sums and counts to be 1D.");
    }
    if (sums.numel() != codebook.numel() || counts.numel() != codebook.numel()) {
        throw std::invalid_argument("Expected sums and counts to match codebook size.");
    }
    if (sums.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected sums to have dtype float32.");
    }
    if (counts.scalar_type() != at::kLong) {
        throw std::invalid_argument("Expected counts to have dtype int64.");
    }

    c10::cuda::CUDAGuard device_guard(grad_flat.device());

    const int64_t num_blocks = grad_flat.numel() / period;
    const dim3 grid(static_cast<unsigned int>(num_blocks));
    const dim3 block(static_cast<unsigned int>(choose_threads(period)));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        grad_flat.scalar_type(),
        "gefen_lloyd_accumulate_cuda",
        [&] {
            gefen_lloyd_accumulate_kernel<scalar_t><<<grid, block>>>(
                grad_flat.data_ptr<scalar_t>(),
                codebook.data_ptr<float>(),
                static_cast<int>(codebook.numel()),
                period,
                num_blocks,
                sums.data_ptr<float>(),
                counts.data_ptr<int64_t>()
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gefen_lloyd_mse_cuda(
    at::Tensor grad_flat,
    at::Tensor old_codebook,
    at::Tensor new_codebook,
    int64_t period,
    at::Tensor mse_normalized_sum,
    at::Tensor mse_original_sum
) {
    validate_common_inputs(grad_flat, old_codebook, period);
    if (!new_codebook.is_cuda()) {
        throw std::invalid_argument("Expected new_codebook to be a CUDA tensor.");
    }
    if (!new_codebook.is_contiguous()) {
        throw std::invalid_argument("Expected new_codebook to be contiguous.");
    }
    if (new_codebook.dim() != 1 || new_codebook.numel() != old_codebook.numel()) {
        throw std::invalid_argument("Expected new_codebook to match old_codebook.");
    }
    if (new_codebook.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected new_codebook to have dtype float32.");
    }
    if (!mse_normalized_sum.is_cuda() || !mse_original_sum.is_cuda()) {
        throw std::invalid_argument("Expected mse accumulators to be CUDA tensors.");
    }
    if (!mse_normalized_sum.is_contiguous() || !mse_original_sum.is_contiguous()) {
        throw std::invalid_argument("Expected mse accumulators to be contiguous.");
    }
    if (mse_normalized_sum.numel() != 1 || mse_original_sum.numel() != 1) {
        throw std::invalid_argument("Expected mse accumulators to be scalar tensors.");
    }
    if (mse_normalized_sum.scalar_type() != at::kFloat || mse_original_sum.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected mse accumulators to have dtype float32.");
    }

    c10::cuda::CUDAGuard device_guard(grad_flat.device());

    const int64_t num_blocks = grad_flat.numel() / period;
    const dim3 grid(static_cast<unsigned int>(num_blocks));
    const dim3 block(static_cast<unsigned int>(choose_threads(period)));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        grad_flat.scalar_type(),
        "gefen_lloyd_mse_cuda",
        [&] {
            gefen_lloyd_mse_kernel<scalar_t><<<grid, block>>>(
                grad_flat.data_ptr<scalar_t>(),
                old_codebook.data_ptr<float>(),
                new_codebook.data_ptr<float>(),
                static_cast<int>(old_codebook.numel()),
                period,
                num_blocks,
                mse_normalized_sum.data_ptr<float>(),
                mse_original_sum.data_ptr<float>()
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
