#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMacros.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <stdexcept>

namespace {

template <typename scalar_t>
__global__ void automatic_vmean_update_kernel(
    float* __restrict__ vmean,
    const scalar_t* __restrict__ grad_view,
    int64_t period,
    int64_t num_blocks,
    float beta2
) {
    extern __shared__ float shared_sum[];

    const int64_t block_idx = static_cast<int64_t>(blockIdx.x);
    if (block_idx >= num_blocks) {
        return;
    }

    const int64_t start = block_idx * period;
    float local_sum = 0.0f;

    for (int64_t offset = threadIdx.x; offset < period; offset += blockDim.x) {
        const float grad_value = static_cast<float>(grad_view[start + offset]);
        local_sum += grad_value * grad_value;
    }

    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float mean_square = shared_sum[0] / static_cast<float>(period);
        const float previous_vmean = vmean[block_idx];
        const float updated_vmean = beta2 * previous_vmean + (1.0f - beta2) * mean_square;
        vmean[block_idx] = updated_vmean;
    }
}

int choose_threads(int64_t period) {
    int threads = 32;
    while (threads < period && threads < 256) {
        threads <<= 1;
    }
    if (threads > 256) {
        threads = 256;
    }
    return threads;
}

}  // namespace

void automatic_vmean_update_cuda(
    at::Tensor vmean,
    at::Tensor grad_view,
    double beta2
) {
    if (!vmean.is_cuda() || !grad_view.is_cuda()) {
        throw std::invalid_argument("Expected vmean and grad_view to be on CUDA.");
    }
    if (!vmean.is_contiguous()) {
        throw std::invalid_argument("Expected vmean to be contiguous.");
    }
    if (!grad_view.is_contiguous()) {
        throw std::invalid_argument("Expected grad_view to be contiguous.");
    }
    if (grad_view.dim() != 2) {
        throw std::invalid_argument("Expected grad_view to be 2D.");
    }
    if (vmean.dim() != 2 || vmean.size(1) != 1) {
        throw std::invalid_argument("Expected vmean to have shape [num_blocks, 1].");
    }
    if (vmean.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected vmean to have dtype float32.");
    }

    c10::cuda::CUDAGuard device_guard(vmean.device());

    const int64_t num_blocks = grad_view.size(0);
    const int64_t period = grad_view.size(1);
    if (vmean.size(0) != num_blocks) {
        throw std::invalid_argument("Expected vmean and grad_view to have the same number of blocks.");
    }
    if (period <= 0) {
        throw std::invalid_argument("Expected grad_view to have a positive period.");
    }

    const int threads = choose_threads(period);
    const dim3 grid(static_cast<unsigned int>(num_blocks));
    const dim3 block(static_cast<unsigned int>(threads));
    const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        grad_view.scalar_type(),
        "automatic_vmean_update_cuda",
        [&] {
            automatic_vmean_update_kernel<scalar_t><<<grid, block, shared_bytes>>>(
                vmean.data_ptr<float>(),
                grad_view.data_ptr<scalar_t>(),
                period,
                num_blocks,
                static_cast<float>(beta2)
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
