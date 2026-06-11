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

__device__ __forceinline__ uint8_t unpack_codebook_index(
    const uint8_t* __restrict__ packed_indices,
    int64_t logical_idx,
    bool packed
) {
    if (!packed) {
        return packed_indices[logical_idx];
    }
    const uint8_t packed_value = packed_indices[logical_idx >> 1];
    if ((logical_idx & 1) == 0) {
        return packed_value & 0x0F;
    }
    return (packed_value >> 4) & 0x0F;
}

__device__ __forceinline__ void store_packed_codebook_index(
    uint8_t* __restrict__ packed_indices,
    int64_t logical_idx,
    uint8_t quantized_index,
    bool packed
) {
    if (!packed) {
        packed_indices[logical_idx] = quantized_index;
        return;
    }

    const int64_t byte_idx = logical_idx >> 1;
    const int nibble_shift = (logical_idx & 1) == 0 ? 0 : 4;
    const uintptr_t raw_address = reinterpret_cast<uintptr_t>(packed_indices + byte_idx);
    const uintptr_t aligned_address = raw_address & ~static_cast<uintptr_t>(0x3);
    unsigned int* word_ptr = reinterpret_cast<unsigned int*>(aligned_address);
    const unsigned int byte_offset = static_cast<unsigned int>(raw_address - aligned_address);
    const unsigned int bit_shift = byte_offset * 8 + static_cast<unsigned int>(nibble_shift);
    const unsigned int nibble_mask = 0xFu << bit_shift;
    const unsigned int nibble_value = (static_cast<unsigned int>(quantized_index) & 0xFu) << bit_shift;

    unsigned int old_word = *word_ptr;
    unsigned int assumed_word = old_word;
    do {
        assumed_word = old_word;
        const unsigned int new_word = (assumed_word & ~nibble_mask) | nibble_value;
        old_word = atomicCAS(word_ptr, assumed_word, new_word);
    } while (old_word != assumed_word);
}

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

template <typename scalar_t, typename magnitude_t>
__global__ void automatic_gefen_fused_update_kernel(
    scalar_t* __restrict__ p,
    const scalar_t* __restrict__ grad_view,
    uint8_t* __restrict__ m_sign,
    magnitude_t* __restrict__ m_magnitude,
    const float* __restrict__ stepsize,
    const float* __restrict__ codebook,
    int codebook_size,
    bool packed_indices,
    int64_t period,
    int64_t num_blocks,
    float beta1,
    float lr
) {
    extern __shared__ float shared_max[];

    const int64_t block_idx = static_cast<int64_t>(blockIdx.x);
    if (block_idx >= num_blocks) {
        return;
    }

    const int64_t start = block_idx * period;
    const float old_magnitude = static_cast<float>(m_magnitude[block_idx]);
    const float step = stepsize[block_idx];
    float local_absmax = 0.0f;

    for (int64_t offset = threadIdx.x; offset < period; offset += blockDim.x) {
        const int64_t idx = start + offset;
        const float coeff = codebook[static_cast<int>(unpack_codebook_index(m_sign, idx, packed_indices))];
        const float current_m = old_magnitude * coeff;
        const float grad_value = static_cast<float>(grad_view[idx]);
        const float updated_value = beta1 * current_m + (1.0f - beta1) * grad_value;
        const float abs_value = fabsf(updated_value);
        if (abs_value > local_absmax) {
            local_absmax = abs_value;
        }
    }

    shared_max[threadIdx.x] = local_absmax;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    const float new_magnitude = shared_max[0];
    if (threadIdx.x == 0) {
        m_magnitude[block_idx] = static_cast<magnitude_t>(new_magnitude);
    }
    __syncthreads();

    for (int64_t offset = threadIdx.x; offset < period; offset += blockDim.x) {
        const int64_t idx = start + offset;
        float normalized_value = 0.0f;
        const float coeff = codebook[static_cast<int>(unpack_codebook_index(m_sign, idx, packed_indices))];
        const float current_m = old_magnitude * coeff;
        const float grad_value = static_cast<float>(grad_view[idx]);
        const float updated_value = beta1 * current_m + (1.0f - beta1) * grad_value;
        if (new_magnitude > 0.0f) {
            normalized_value = updated_value / new_magnitude;
        }
        const uint8_t quantized_index = nearest_codebook_index(normalized_value, codebook, codebook_size);
        store_packed_codebook_index(m_sign, idx, quantized_index, packed_indices);
        if (lr != 0.0f) {
            const float quantized_value = codebook[static_cast<int>(quantized_index)] * new_magnitude;
            const float update_value = quantized_value * step * lr;
            p[idx] = static_cast<scalar_t>(static_cast<float>(p[idx]) - update_value);
        }
    }
}

template <typename scalar_t, typename magnitude_t>
__global__ void automatic_gefen_muon_momentum_kernel(
    const scalar_t* __restrict__ grad_view,
    uint8_t* __restrict__ m_sign,
    magnitude_t* __restrict__ m_magnitude,
    const float* __restrict__ codebook,
    scalar_t* __restrict__ momentum_update,
    int codebook_size,
    bool packed_indices,
    int64_t period,
    int64_t num_blocks,
    float beta1
) {
    extern __shared__ float shared_max[];

    const int64_t block_idx = static_cast<int64_t>(blockIdx.x);
    if (block_idx >= num_blocks) {
        return;
    }

    const int64_t start = block_idx * period;
    const float old_magnitude = static_cast<float>(m_magnitude[block_idx]);
    float local_absmax = 0.0f;

    for (int64_t offset = threadIdx.x; offset < period; offset += blockDim.x) {
        const int64_t idx = start + offset;
        const float coeff = codebook[static_cast<int>(unpack_codebook_index(m_sign, idx, packed_indices))];
        const float current_m = old_magnitude * coeff;
        const float grad_value = static_cast<float>(grad_view[idx]);
        const float updated_value = beta1 * current_m + (1.0f - beta1) * grad_value;
        const float abs_value = fabsf(updated_value);
        if (abs_value > local_absmax) {
            local_absmax = abs_value;
        }
    }

    shared_max[threadIdx.x] = local_absmax;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    const float new_magnitude = shared_max[0];
    if (threadIdx.x == 0) {
        m_magnitude[block_idx] = static_cast<magnitude_t>(new_magnitude);
    }
    __syncthreads();

    for (int64_t offset = threadIdx.x; offset < period; offset += blockDim.x) {
        const int64_t idx = start + offset;
        const float coeff = codebook[static_cast<int>(unpack_codebook_index(m_sign, idx, packed_indices))];
        const float current_m = old_magnitude * coeff;
        const float grad_value = static_cast<float>(grad_view[idx]);
        const float updated_value = beta1 * current_m + (1.0f - beta1) * grad_value;
        float normalized_value = 0.0f;
        if (new_magnitude > 0.0f) {
            normalized_value = updated_value / new_magnitude;
        }
        const uint8_t quantized_index = nearest_codebook_index(normalized_value, codebook, codebook_size);
        store_packed_codebook_index(m_sign, idx, quantized_index, packed_indices);
        momentum_update[idx] = static_cast<scalar_t>(codebook[static_cast<int>(quantized_index)] * new_magnitude);
    }
}

template <typename scalar_t, typename magnitude_t, typename vmean_t>
__global__ void automatic_gefen_fused_update_from_vmean_kernel(
    scalar_t* __restrict__ p,
    const scalar_t* __restrict__ grad_view,
    uint8_t* __restrict__ m_sign,
    magnitude_t* __restrict__ m_magnitude,
    const vmean_t* __restrict__ vmean,
    const float* __restrict__ codebook,
    int codebook_size,
    bool packed_indices,
    int64_t period,
    int64_t num_blocks,
    float beta1,
    float lr,
    float bias_correction_1,
    float bias_correction_2,
    float eps
) {
    extern __shared__ float shared_max[];

    const int64_t block_idx = static_cast<int64_t>(blockIdx.x);
    if (block_idx >= num_blocks) {
        return;
    }

    const int64_t start = block_idx * period;
    const float old_magnitude = static_cast<float>(m_magnitude[block_idx]);
    const float corrected_vmean = static_cast<float>(vmean[block_idx]) / bias_correction_2;
    const float step = (1.0f / bias_correction_1) / (sqrtf(corrected_vmean) + eps);
    float local_absmax = 0.0f;

    for (int64_t offset = threadIdx.x; offset < period; offset += blockDim.x) {
        const int64_t idx = start + offset;
        const float coeff = codebook[static_cast<int>(unpack_codebook_index(m_sign, idx, packed_indices))];
        const float current_m = old_magnitude * coeff;
        const float grad_value = static_cast<float>(grad_view[idx]);
        const float updated_value = beta1 * current_m + (1.0f - beta1) * grad_value;
        const float abs_value = fabsf(updated_value);
        if (abs_value > local_absmax) {
            local_absmax = abs_value;
        }
    }

    shared_max[threadIdx.x] = local_absmax;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    const float new_magnitude = shared_max[0];
    if (threadIdx.x == 0) {
        m_magnitude[block_idx] = static_cast<magnitude_t>(new_magnitude);
    }
    __syncthreads();

    for (int64_t offset = threadIdx.x; offset < period; offset += blockDim.x) {
        const int64_t idx = start + offset;
        float normalized_value = 0.0f;
        const float coeff = codebook[static_cast<int>(unpack_codebook_index(m_sign, idx, packed_indices))];
        const float current_m = old_magnitude * coeff;
        const float grad_value = static_cast<float>(grad_view[idx]);
        const float updated_value = beta1 * current_m + (1.0f - beta1) * grad_value;
        if (new_magnitude > 0.0f) {
            normalized_value = updated_value / new_magnitude;
        }
        const uint8_t quantized_index = nearest_codebook_index(normalized_value, codebook, codebook_size);
        store_packed_codebook_index(m_sign, idx, quantized_index, packed_indices);
        if (lr != 0.0f) {
            const float quantized_value = codebook[static_cast<int>(quantized_index)] * new_magnitude;
            const float update_value = quantized_value * step * lr;
            p[idx] = static_cast<scalar_t>(static_cast<float>(p[idx]) - update_value);
        }
    }
}

template <typename scalar_t, typename magnitude_t, typename vmean_t>
__global__ void automatic_gefen_period1_update_kernel(
    scalar_t* __restrict__ p,
    const scalar_t* __restrict__ grad_flat,
    uint8_t* __restrict__ m_sign,
    magnitude_t* __restrict__ m_magnitude,
    vmean_t* __restrict__ vmean,
    const float* __restrict__ codebook,
    int codebook_size,
    int64_t total_numel,
    int64_t period_one_divisor,
    int64_t num_magnitude_blocks,
    float beta1,
    float beta2,
    float lr,
    float bias_correction_1,
    float bias_correction_2,
    float eps
) {
    extern __shared__ float shared_max[];

    const int64_t block_idx = static_cast<int64_t>(blockIdx.x);
    if (block_idx >= num_magnitude_blocks) {
        return;
    }

    const int64_t start = block_idx * period_one_divisor;
    const float old_magnitude = static_cast<float>(m_magnitude[block_idx]);
    float local_absmax = 0.0f;

    for (int64_t offset = threadIdx.x; offset < period_one_divisor; offset += blockDim.x) {
        const int64_t idx = start + offset;
        if (idx < total_numel) {
            const float grad_value = static_cast<float>(grad_flat[idx]);
            const float previous_vmean = static_cast<float>(vmean[idx]);
            const float updated_vmean = beta2 * previous_vmean + (1.0f - beta2) * grad_value * grad_value;
            vmean[idx] = static_cast<vmean_t>(updated_vmean);

            const float coeff = codebook[static_cast<int>(m_sign[idx])];
            const float current_m = old_magnitude * coeff;
            const float updated_m = beta1 * current_m + (1.0f - beta1) * grad_value;
            const float abs_value = fabsf(updated_m);
            if (abs_value > local_absmax) {
                local_absmax = abs_value;
            }
        }
    }

    shared_max[threadIdx.x] = local_absmax;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    const float new_magnitude = shared_max[0];
    if (threadIdx.x == 0) {
        m_magnitude[block_idx] = static_cast<magnitude_t>(new_magnitude);
    }
    __syncthreads();

    for (int64_t offset = threadIdx.x; offset < period_one_divisor; offset += blockDim.x) {
        const int64_t idx = start + offset;
        if (idx < total_numel) {
            const float grad_value = static_cast<float>(grad_flat[idx]);
            const float coeff = codebook[static_cast<int>(m_sign[idx])];
            const float current_m = old_magnitude * coeff;
            const float updated_m = beta1 * current_m + (1.0f - beta1) * grad_value;
            float normalized_value = 0.0f;
            if (new_magnitude > 0.0f) {
                normalized_value = updated_m / new_magnitude;
            }
            const uint8_t quantized_index = nearest_codebook_index(normalized_value, codebook, codebook_size);
            m_sign[idx] = quantized_index;

            const float corrected_vmean = static_cast<float>(vmean[idx]) / bias_correction_2;
            const float step = (1.0f / bias_correction_1) / (sqrtf(corrected_vmean) + eps);
            const float quantized_value = codebook[static_cast<int>(quantized_index)] * new_magnitude;
            const float update_value = quantized_value * step * lr;
            p[idx] = static_cast<scalar_t>(static_cast<float>(p[idx]) - update_value);
        }
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

template <typename scalar_t, typename magnitude_t>
void launch_automatic_gefen_fused_update(
    at::Tensor p,
    at::Tensor grad_view,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor stepsize,
    at::Tensor codebook,
    bool packed_indices,
    int64_t period,
    int64_t num_blocks,
    int threads,
    size_t shared_bytes,
    double beta1,
    double lr
) {
    const dim3 grid(static_cast<unsigned int>(num_blocks));
    const dim3 block(static_cast<unsigned int>(threads));
    automatic_gefen_fused_update_kernel<scalar_t, magnitude_t><<<grid, block, shared_bytes>>>(
        p.data_ptr<scalar_t>(),
        grad_view.data_ptr<scalar_t>(),
        m_sign.data_ptr<uint8_t>(),
        m_magnitude.data_ptr<magnitude_t>(),
        stepsize.data_ptr<float>(),
        codebook.data_ptr<float>(),
        static_cast<int>(codebook.numel()),
        packed_indices,
        period,
        num_blocks,
        static_cast<float>(beta1),
        static_cast<float>(lr)
    );
}

template <typename scalar_t, typename magnitude_t, typename vmean_t>
void launch_automatic_gefen_fused_update_from_vmean(
    at::Tensor p,
    at::Tensor grad_view,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor vmean,
    at::Tensor codebook,
    bool packed_indices,
    int64_t period,
    int64_t num_blocks,
    int threads,
    size_t shared_bytes,
    double beta1,
    double lr,
    double bias_correction_1,
    double bias_correction_2,
    double eps
) {
    const dim3 grid(static_cast<unsigned int>(num_blocks));
    const dim3 block(static_cast<unsigned int>(threads));
    automatic_gefen_fused_update_from_vmean_kernel<scalar_t, magnitude_t, vmean_t><<<grid, block, shared_bytes>>>(
        p.data_ptr<scalar_t>(),
        grad_view.data_ptr<scalar_t>(),
        m_sign.data_ptr<uint8_t>(),
        m_magnitude.data_ptr<magnitude_t>(),
        vmean.data_ptr<vmean_t>(),
        codebook.data_ptr<float>(),
        static_cast<int>(codebook.numel()),
        packed_indices,
        period,
        num_blocks,
        static_cast<float>(beta1),
        static_cast<float>(lr),
        static_cast<float>(bias_correction_1),
        static_cast<float>(bias_correction_2),
        static_cast<float>(eps)
    );
}

template <typename scalar_t, typename magnitude_t, typename vmean_t>
void launch_automatic_gefen_period1_update(
    at::Tensor p,
    at::Tensor grad_flat,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor vmean,
    at::Tensor codebook,
    int64_t total_numel,
    int64_t period_one_divisor,
    int64_t num_magnitude_blocks,
    double beta1,
    double beta2,
    double lr,
    double bias_correction_1,
    double bias_correction_2,
    double eps
) {
    const int threads = choose_threads(period_one_divisor);
    const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);
    const dim3 block(static_cast<unsigned int>(threads));
    const dim3 grid(static_cast<unsigned int>(num_magnitude_blocks));
    automatic_gefen_period1_update_kernel<scalar_t, magnitude_t, vmean_t><<<grid, block, shared_bytes>>>(
        p.data_ptr<scalar_t>(),
        grad_flat.data_ptr<scalar_t>(),
        m_sign.data_ptr<uint8_t>(),
        m_magnitude.data_ptr<magnitude_t>(),
        vmean.data_ptr<vmean_t>(),
        codebook.data_ptr<float>(),
        static_cast<int>(codebook.numel()),
        total_numel,
        period_one_divisor,
        num_magnitude_blocks,
        static_cast<float>(beta1),
        static_cast<float>(beta2),
        static_cast<float>(lr),
        static_cast<float>(bias_correction_1),
        static_cast<float>(bias_correction_2),
        static_cast<float>(eps)
    );
}

template <typename scalar_t, typename magnitude_t>
void launch_automatic_gefen_muon_momentum(
    at::Tensor grad_view,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor codebook,
    at::Tensor momentum_update,
    bool packed_indices,
    int64_t period,
    int64_t num_blocks,
    int threads,
    size_t shared_bytes,
    double beta1
) {
    const dim3 grid(static_cast<unsigned int>(num_blocks));
    const dim3 block(static_cast<unsigned int>(threads));
    automatic_gefen_muon_momentum_kernel<scalar_t, magnitude_t><<<grid, block, shared_bytes>>>(
        grad_view.data_ptr<scalar_t>(),
        m_sign.data_ptr<uint8_t>(),
        m_magnitude.data_ptr<magnitude_t>(),
        codebook.data_ptr<float>(),
        momentum_update.data_ptr<scalar_t>(),
        static_cast<int>(codebook.numel()),
        packed_indices,
        period,
        num_blocks,
        static_cast<float>(beta1)
    );
}

}  // namespace

void automatic_gefen_fused_update_cuda(
    at::Tensor p,
    at::Tensor grad_view,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor stepsize,
    at::Tensor codebook,
    bool packed_indices,
    double beta1,
    double lr
) {
    if (!p.is_cuda() || !grad_view.is_cuda() || !m_sign.is_cuda() || !m_magnitude.is_cuda() || !stepsize.is_cuda() || !codebook.is_cuda()) {
        throw std::invalid_argument("Expected all tensors to be on CUDA.");
    }
    if (!p.is_contiguous()) {
        throw std::invalid_argument("Expected p to be contiguous.");
    }
    if (!grad_view.is_contiguous()) {
        throw std::invalid_argument("Expected grad_view to be contiguous.");
    }
    if (!m_sign.is_contiguous()) {
        throw std::invalid_argument("Expected m_sign to be contiguous.");
    }
    if (!m_magnitude.is_contiguous()) {
        throw std::invalid_argument("Expected m_magnitude to be contiguous.");
    }
    if (!stepsize.is_contiguous()) {
        throw std::invalid_argument("Expected stepsize to be contiguous.");
    }
    if (!codebook.is_contiguous()) {
        throw std::invalid_argument("Expected codebook to be contiguous.");
    }
    if (grad_view.dim() != 2) {
        throw std::invalid_argument("Expected grad_view to be 2D.");
    }
    if (grad_view.scalar_type() != p.scalar_type()) {
        throw std::invalid_argument("Expected grad_view dtype to match p dtype.");
    }
    if (m_magnitude.dim() != 2 || m_magnitude.size(1) != 1) {
        throw std::invalid_argument("Expected m_magnitude to have shape [num_blocks, 1].");
    }
    if (stepsize.dim() != 2 || stepsize.size(1) != 1) {
        throw std::invalid_argument("Expected stepsize to have shape [num_blocks, 1].");
    }
    if (m_sign.scalar_type() != at::kByte) {
        throw std::invalid_argument("Expected m_sign to have dtype uint8.");
    }
    if (m_magnitude.scalar_type() != at::kFloat && m_magnitude.scalar_type() != at::kHalf && m_magnitude.scalar_type() != at::kBFloat16) {
        throw std::invalid_argument("Expected m_magnitude to have dtype float32, float16, or bfloat16.");
    }
    if (stepsize.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected stepsize to have dtype float32.");
    }
    if (codebook.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected codebook to have dtype float32.");
    }

    c10::cuda::CUDAGuard device_guard(p.device());

    const int64_t num_blocks = grad_view.size(0);
    const int64_t period = grad_view.size(1);
    const int64_t total_numel = num_blocks * period;
    if (p.numel() != total_numel) {
        throw std::invalid_argument("Expected p.numel() to match grad_view.numel().");
    }
    if (packed_indices) {
        if (m_sign.numel() != (total_numel + 1) / 2) {
            throw std::invalid_argument("Expected packed m_sign.numel() to be ceil(total_numel / 2).");
        }
    } else if (!packed_indices && m_sign.numel() != total_numel) {
        throw std::invalid_argument("Expected unpacked m_sign.numel() to match grad_view.numel().");
    }
    if (m_magnitude.size(0) != num_blocks || stepsize.size(0) != num_blocks) {
        throw std::invalid_argument("Expected m_magnitude and stepsize to match the number of blocks.");
    }
    // these two may create additional memory footprint.
    const int threads = choose_threads(period);
    const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        p.scalar_type(),
        "automatic_gefen_fused_update_cuda",
        [&] {
            if (m_magnitude.scalar_type() == at::kFloat) {
                launch_automatic_gefen_fused_update<scalar_t, float>(
                    p, grad_view, m_sign, m_magnitude, stepsize, codebook,
                    packed_indices, period, num_blocks, threads, shared_bytes, beta1, lr
                );
            } else if (m_magnitude.scalar_type() == at::kHalf) {
                launch_automatic_gefen_fused_update<scalar_t, at::Half>(
                    p, grad_view, m_sign, m_magnitude, stepsize, codebook,
                    packed_indices, period, num_blocks, threads, shared_bytes, beta1, lr
                );
            } else if (m_magnitude.scalar_type() == at::kBFloat16) {
                launch_automatic_gefen_fused_update<scalar_t, at::BFloat16>(
                    p, grad_view, m_sign, m_magnitude, stepsize, codebook,
                    packed_indices, period, num_blocks, threads, shared_bytes, beta1, lr
                );
            } else {
                throw std::invalid_argument("Expected m_magnitude to have dtype float32, float16, or bfloat16.");
            }
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void automatic_gefen_fused_update_from_vmean_cuda(
    at::Tensor p,
    at::Tensor grad_view,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor vmean,
    at::Tensor codebook,
    bool packed_indices,
    double beta1,
    double lr,
    double bias_correction_1,
    double bias_correction_2,
    double eps
) {
    if (!p.is_cuda() || !grad_view.is_cuda() || !m_sign.is_cuda() || !m_magnitude.is_cuda() || !vmean.is_cuda() || !codebook.is_cuda()) {
        throw std::invalid_argument("Expected all tensors to be on CUDA.");
    }
    if (!p.is_contiguous()) {
        throw std::invalid_argument("Expected p to be contiguous.");
    }
    if (!grad_view.is_contiguous()) {
        throw std::invalid_argument("Expected grad_view to be contiguous.");
    }
    if (!m_sign.is_contiguous()) {
        throw std::invalid_argument("Expected m_sign to be contiguous.");
    }
    if (!m_magnitude.is_contiguous()) {
        throw std::invalid_argument("Expected m_magnitude to be contiguous.");
    }
    if (!vmean.is_contiguous()) {
        throw std::invalid_argument("Expected vmean to be contiguous.");
    }
    if (!codebook.is_contiguous()) {
        throw std::invalid_argument("Expected codebook to be contiguous.");
    }
    if (grad_view.dim() != 2) {
        throw std::invalid_argument("Expected grad_view to be 2D.");
    }
    if (grad_view.scalar_type() != p.scalar_type()) {
        throw std::invalid_argument("Expected grad_view dtype to match p dtype.");
    }
    if (m_magnitude.dim() != 2 || m_magnitude.size(1) != 1) {
        throw std::invalid_argument("Expected m_magnitude to have shape [num_blocks, 1].");
    }
    if (vmean.dim() != 2 || vmean.size(1) != 1) {
        throw std::invalid_argument("Expected vmean to have shape [num_blocks, 1].");
    }
    if (m_sign.scalar_type() != at::kByte) {
        throw std::invalid_argument("Expected m_sign to have dtype uint8.");
    }
    if (m_magnitude.scalar_type() != at::kFloat && m_magnitude.scalar_type() != at::kHalf && m_magnitude.scalar_type() != at::kBFloat16) {
        throw std::invalid_argument("Expected m_magnitude to have dtype float32, float16, or bfloat16.");
    }
    if (vmean.scalar_type() != at::kFloat && vmean.scalar_type() != at::kHalf && vmean.scalar_type() != at::kBFloat16) {
        throw std::invalid_argument("Expected vmean to have dtype float32, float16, or bfloat16.");
    }
    if (codebook.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected codebook to have dtype float32.");
    }
    if (bias_correction_1 <= 0.0) {
        throw std::invalid_argument("Expected bias_correction_1 to be positive.");
    }
    if (bias_correction_2 <= 0.0) {
        throw std::invalid_argument("Expected bias_correction_2 to be positive.");
    }
    if (eps < 0.0) {
        throw std::invalid_argument("Expected eps to be non-negative.");
    }

    c10::cuda::CUDAGuard device_guard(p.device());

    const int64_t num_blocks = grad_view.size(0);
    const int64_t period = grad_view.size(1);
    const int64_t total_numel = num_blocks * period;
    if (p.numel() != total_numel) {
        throw std::invalid_argument("Expected p.numel() to match grad_view.numel().");
    }
    if (packed_indices) {
        if (m_sign.numel() != (total_numel + 1) / 2) {
            throw std::invalid_argument("Expected packed m_sign.numel() to be ceil(total_numel / 2).");
        }
    } else if (!packed_indices && m_sign.numel() != total_numel) {
        throw std::invalid_argument("Expected unpacked m_sign.numel() to match grad_view.numel().");
    }
    if (m_magnitude.size(0) != num_blocks || vmean.size(0) != num_blocks) {
        throw std::invalid_argument("Expected m_magnitude and vmean to match the number of blocks.");
    }

    const int threads = choose_threads(period);
    const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        p.scalar_type(),
        "automatic_gefen_fused_update_from_vmean_cuda",
        [&] {
            if (m_magnitude.scalar_type() == at::kFloat && vmean.scalar_type() == at::kFloat) {
                launch_automatic_gefen_fused_update_from_vmean<scalar_t, float, float>(
                    p, grad_view, m_sign, m_magnitude, vmean, codebook,
                    packed_indices, period, num_blocks, threads, shared_bytes,
                    beta1, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kFloat && vmean.scalar_type() == at::kHalf) {
                launch_automatic_gefen_fused_update_from_vmean<scalar_t, float, at::Half>(
                    p, grad_view, m_sign, m_magnitude, vmean, codebook,
                    packed_indices, period, num_blocks, threads, shared_bytes,
                    beta1, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kFloat && vmean.scalar_type() == at::kBFloat16) {
                launch_automatic_gefen_fused_update_from_vmean<scalar_t, float, at::BFloat16>(
                    p, grad_view, m_sign, m_magnitude, vmean, codebook,
                    packed_indices, period, num_blocks, threads, shared_bytes,
                    beta1, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kHalf && vmean.scalar_type() == at::kFloat) {
                launch_automatic_gefen_fused_update_from_vmean<scalar_t, at::Half, float>(
                    p, grad_view, m_sign, m_magnitude, vmean, codebook,
                    packed_indices, period, num_blocks, threads, shared_bytes,
                    beta1, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kHalf && vmean.scalar_type() == at::kHalf) {
                launch_automatic_gefen_fused_update_from_vmean<scalar_t, at::Half, at::Half>(
                    p, grad_view, m_sign, m_magnitude, vmean, codebook,
                    packed_indices, period, num_blocks, threads, shared_bytes,
                    beta1, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kHalf && vmean.scalar_type() == at::kBFloat16) {
                launch_automatic_gefen_fused_update_from_vmean<scalar_t, at::Half, at::BFloat16>(
                    p, grad_view, m_sign, m_magnitude, vmean, codebook,
                    packed_indices, period, num_blocks, threads, shared_bytes,
                    beta1, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kBFloat16 && vmean.scalar_type() == at::kFloat) {
                launch_automatic_gefen_fused_update_from_vmean<scalar_t, at::BFloat16, float>(
                    p, grad_view, m_sign, m_magnitude, vmean, codebook,
                    packed_indices, period, num_blocks, threads, shared_bytes,
                    beta1, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kBFloat16 && vmean.scalar_type() == at::kHalf) {
                launch_automatic_gefen_fused_update_from_vmean<scalar_t, at::BFloat16, at::Half>(
                    p, grad_view, m_sign, m_magnitude, vmean, codebook,
                    packed_indices, period, num_blocks, threads, shared_bytes,
                    beta1, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kBFloat16 && vmean.scalar_type() == at::kBFloat16) {
                launch_automatic_gefen_fused_update_from_vmean<scalar_t, at::BFloat16, at::BFloat16>(
                    p, grad_view, m_sign, m_magnitude, vmean, codebook,
                    packed_indices, period, num_blocks, threads, shared_bytes,
                    beta1, lr, bias_correction_1, bias_correction_2, eps
                );
            } else {
                throw std::invalid_argument("Expected m_magnitude and vmean to have dtype float32, float16, or bfloat16.");
            }
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void automatic_gefen_period1_update_cuda(
    at::Tensor p,
    at::Tensor grad_flat,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor vmean,
    at::Tensor codebook,
    int64_t period_one_divisor,
    double beta1,
    double beta2,
    double lr,
    double bias_correction_1,
    double bias_correction_2,
    double eps
) {
    if (!p.is_cuda() || !grad_flat.is_cuda() || !m_sign.is_cuda() || !m_magnitude.is_cuda() || !vmean.is_cuda() || !codebook.is_cuda()) {
        throw std::invalid_argument("Expected all tensors to be on CUDA.");
    }
    if (!p.is_contiguous()) {
        throw std::invalid_argument("Expected p to be contiguous.");
    }
    if (!grad_flat.is_contiguous()) {
        throw std::invalid_argument("Expected grad_flat to be contiguous.");
    }
    if (!m_sign.is_contiguous()) {
        throw std::invalid_argument("Expected m_sign to be contiguous.");
    }
    if (!m_magnitude.is_contiguous()) {
        throw std::invalid_argument("Expected m_magnitude to be contiguous.");
    }
    if (!vmean.is_contiguous()) {
        throw std::invalid_argument("Expected vmean to be contiguous.");
    }
    if (!codebook.is_contiguous()) {
        throw std::invalid_argument("Expected codebook to be contiguous.");
    }
    if (grad_flat.dim() != 1) {
        throw std::invalid_argument("Expected grad_flat to be 1D.");
    }
    if (grad_flat.scalar_type() != p.scalar_type()) {
        throw std::invalid_argument("Expected grad_flat dtype to match p dtype.");
    }
    if (m_sign.scalar_type() != at::kByte) {
        throw std::invalid_argument("Expected m_sign to have dtype uint8.");
    }
    if (m_magnitude.dim() != 2 || m_magnitude.size(1) != 1) {
        throw std::invalid_argument("Expected m_magnitude to have shape [num_magnitude_blocks, 1].");
    }
    if (vmean.dim() != 2 || vmean.size(1) != 1) {
        throw std::invalid_argument("Expected vmean to have shape [numel, 1].");
    }
    if (m_magnitude.scalar_type() != at::kFloat && m_magnitude.scalar_type() != at::kHalf && m_magnitude.scalar_type() != at::kBFloat16) {
        throw std::invalid_argument("Expected m_magnitude to have dtype float32, float16, or bfloat16.");
    }
    if (vmean.scalar_type() != at::kFloat && vmean.scalar_type() != at::kHalf && vmean.scalar_type() != at::kBFloat16) {
        throw std::invalid_argument("Expected vmean to have dtype float32, float16, or bfloat16.");
    }
    if (codebook.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected codebook to have dtype float32.");
    }
    if (codebook.numel() <= 0 || codebook.numel() > 256) {
        throw std::invalid_argument("Expected codebook size in [1, 256].");
    }
    if (period_one_divisor <= 0) {
        throw std::invalid_argument("Expected period_one_divisor to be positive.");
    }
    if (beta2 < 0.0 || beta2 > 1.0) {
        throw std::invalid_argument("Expected beta2 to be in [0, 1].");
    }
    if (bias_correction_1 <= 0.0) {
        throw std::invalid_argument("Expected bias_correction_1 to be positive.");
    }
    if (bias_correction_2 <= 0.0) {
        throw std::invalid_argument("Expected bias_correction_2 to be positive.");
    }
    if (eps < 0.0) {
        throw std::invalid_argument("Expected eps to be non-negative.");
    }

    c10::cuda::CUDAGuard device_guard(p.device());

    const int64_t total_numel = grad_flat.numel();
    if (p.numel() != total_numel) {
        throw std::invalid_argument("Expected p.numel() to match grad_flat.numel().");
    }
    if (total_numel % period_one_divisor != 0) {
        throw std::invalid_argument("Expected period_one_divisor to divide grad_flat.numel().");
    }
    if (m_sign.numel() != total_numel) {
        throw std::invalid_argument("Expected m_sign.numel() to match grad_flat.numel().");
    }
    const int64_t num_magnitude_blocks = total_numel / period_one_divisor;
    if (m_magnitude.size(0) != num_magnitude_blocks) {
        throw std::invalid_argument("Expected m_magnitude to have one row per period-one magnitude block.");
    }
    if (vmean.size(0) != total_numel) {
        throw std::invalid_argument("Expected vmean to have one row per gradient element.");
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        p.scalar_type(),
        "automatic_gefen_period1_update_cuda",
        [&] {
            if (m_magnitude.scalar_type() == at::kFloat && vmean.scalar_type() == at::kFloat) {
                launch_automatic_gefen_period1_update<scalar_t, float, float>(
                    p, grad_flat, m_sign, m_magnitude, vmean, codebook, total_numel,
                    period_one_divisor, num_magnitude_blocks,
                    beta1, beta2, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kFloat && vmean.scalar_type() == at::kHalf) {
                launch_automatic_gefen_period1_update<scalar_t, float, at::Half>(
                    p, grad_flat, m_sign, m_magnitude, vmean, codebook, total_numel,
                    period_one_divisor, num_magnitude_blocks,
                    beta1, beta2, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kFloat && vmean.scalar_type() == at::kBFloat16) {
                launch_automatic_gefen_period1_update<scalar_t, float, at::BFloat16>(
                    p, grad_flat, m_sign, m_magnitude, vmean, codebook, total_numel,
                    period_one_divisor, num_magnitude_blocks,
                    beta1, beta2, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kHalf && vmean.scalar_type() == at::kFloat) {
                launch_automatic_gefen_period1_update<scalar_t, at::Half, float>(
                    p, grad_flat, m_sign, m_magnitude, vmean, codebook, total_numel,
                    period_one_divisor, num_magnitude_blocks,
                    beta1, beta2, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kHalf && vmean.scalar_type() == at::kHalf) {
                launch_automatic_gefen_period1_update<scalar_t, at::Half, at::Half>(
                    p, grad_flat, m_sign, m_magnitude, vmean, codebook, total_numel,
                    period_one_divisor, num_magnitude_blocks,
                    beta1, beta2, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kHalf && vmean.scalar_type() == at::kBFloat16) {
                launch_automatic_gefen_period1_update<scalar_t, at::Half, at::BFloat16>(
                    p, grad_flat, m_sign, m_magnitude, vmean, codebook, total_numel,
                    period_one_divisor, num_magnitude_blocks,
                    beta1, beta2, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kBFloat16 && vmean.scalar_type() == at::kFloat) {
                launch_automatic_gefen_period1_update<scalar_t, at::BFloat16, float>(
                    p, grad_flat, m_sign, m_magnitude, vmean, codebook, total_numel,
                    period_one_divisor, num_magnitude_blocks,
                    beta1, beta2, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kBFloat16 && vmean.scalar_type() == at::kHalf) {
                launch_automatic_gefen_period1_update<scalar_t, at::BFloat16, at::Half>(
                    p, grad_flat, m_sign, m_magnitude, vmean, codebook, total_numel,
                    period_one_divisor, num_magnitude_blocks,
                    beta1, beta2, lr, bias_correction_1, bias_correction_2, eps
                );
            } else if (m_magnitude.scalar_type() == at::kBFloat16 && vmean.scalar_type() == at::kBFloat16) {
                launch_automatic_gefen_period1_update<scalar_t, at::BFloat16, at::BFloat16>(
                    p, grad_flat, m_sign, m_magnitude, vmean, codebook, total_numel,
                    period_one_divisor, num_magnitude_blocks,
                    beta1, beta2, lr, bias_correction_1, bias_correction_2, eps
                );
            } else {
                throw std::invalid_argument("Expected m_magnitude and vmean to have dtype float32, float16, or bfloat16.");
            }
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void automatic_gefen_muon_momentum_cuda(
    at::Tensor grad_view,
    at::Tensor m_sign,
    at::Tensor m_magnitude,
    at::Tensor codebook,
    at::Tensor momentum_update,
    bool packed_indices,
    double beta1
) {
    if (!grad_view.is_cuda() || !m_sign.is_cuda() || !m_magnitude.is_cuda() || !codebook.is_cuda() || !momentum_update.is_cuda()) {
        throw std::invalid_argument("Expected all tensors to be on CUDA.");
    }
    if (!grad_view.is_contiguous()) {
        throw std::invalid_argument("Expected grad_view to be contiguous.");
    }
    if (!m_sign.is_contiguous()) {
        throw std::invalid_argument("Expected m_sign to be contiguous.");
    }
    if (!m_magnitude.is_contiguous()) {
        throw std::invalid_argument("Expected m_magnitude to be contiguous.");
    }
    if (!codebook.is_contiguous()) {
        throw std::invalid_argument("Expected codebook to be contiguous.");
    }
    if (!momentum_update.is_contiguous()) {
        throw std::invalid_argument("Expected momentum_update to be contiguous.");
    }
    if (grad_view.dim() != 2) {
        throw std::invalid_argument("Expected grad_view to be 2D.");
    }
    if (momentum_update.dim() != grad_view.dim() || momentum_update.size(0) != grad_view.size(0) || momentum_update.size(1) != grad_view.size(1)) {
        throw std::invalid_argument("Expected momentum_update shape to match grad_view shape.");
    }
    if (momentum_update.scalar_type() != grad_view.scalar_type()) {
        throw std::invalid_argument("Expected momentum_update dtype to match grad_view dtype.");
    }
    if (m_sign.scalar_type() != at::kByte) {
        throw std::invalid_argument("Expected m_sign to have dtype uint8.");
    }
    if (m_magnitude.dim() != 2 || m_magnitude.size(1) != 1) {
        throw std::invalid_argument("Expected m_magnitude to have shape [num_blocks, 1].");
    }
    if (m_magnitude.scalar_type() != at::kFloat && m_magnitude.scalar_type() != at::kHalf && m_magnitude.scalar_type() != at::kBFloat16) {
        throw std::invalid_argument("Expected m_magnitude to have dtype float32, float16, or bfloat16.");
    }
    if (codebook.scalar_type() != at::kFloat) {
        throw std::invalid_argument("Expected codebook to have dtype float32.");
    }
    if (codebook.numel() <= 0 || codebook.numel() > 256) {
        throw std::invalid_argument("Expected codebook size in [1, 256].");
    }
    if (beta1 < 0.0 || beta1 >= 1.0) {
        throw std::invalid_argument("Expected beta1 to be in [0, 1).");
    }

    c10::cuda::CUDAGuard device_guard(grad_view.device());

    const int64_t num_blocks = grad_view.size(0);
    const int64_t period = grad_view.size(1);
    const int64_t total_numel = num_blocks * period;
    if (packed_indices) {
        if (m_sign.numel() != (total_numel + 1) / 2) {
            throw std::invalid_argument("Expected packed m_sign.numel() to be ceil(total_numel / 2).");
        }
    } else if (!packed_indices && m_sign.numel() != total_numel) {
        throw std::invalid_argument("Expected unpacked m_sign.numel() to match grad_view.numel().");
    }
    if (m_magnitude.size(0) != num_blocks) {
        throw std::invalid_argument("Expected m_magnitude to match the number of blocks.");
    }

    const int threads = choose_threads(period);
    const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        grad_view.scalar_type(),
        "automatic_gefen_muon_momentum_cuda",
        [&] {
            if (m_magnitude.scalar_type() == at::kFloat) {
                launch_automatic_gefen_muon_momentum<scalar_t, float>(
                    grad_view, m_sign, m_magnitude, codebook, momentum_update,
                    packed_indices, period, num_blocks, threads, shared_bytes, beta1
                );
            } else if (m_magnitude.scalar_type() == at::kHalf) {
                launch_automatic_gefen_muon_momentum<scalar_t, at::Half>(
                    grad_view, m_sign, m_magnitude, codebook, momentum_update,
                    packed_indices, period, num_blocks, threads, shared_bytes, beta1
                );
            } else if (m_magnitude.scalar_type() == at::kBFloat16) {
                launch_automatic_gefen_muon_momentum<scalar_t, at::BFloat16>(
                    grad_view, m_sign, m_magnitude, codebook, momentum_update,
                    packed_indices, period, num_blocks, threads, shared_bytes, beta1
                );
            } else {
                throw std::invalid_argument("Expected m_magnitude to have dtype float32, float16, or bfloat16.");
            }
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
