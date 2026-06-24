#include <torch/extension.h>

void gefen_exact_histogram_cuda(
    at::Tensor grad_flat,
    int64_t period,
    at::Tensor bin_counts
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gefen_exact_histogram_cuda",
        &gefen_exact_histogram_cuda,
        "Accumulate exact-DP histogram counts from raw gradients on CUDA"
    );
}
