#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

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
);

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
);

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
);

void automatic_vmean_update_cuda(
    at::Tensor vmean,
    at::Tensor grad_view,
    double beta2
);

void gefen_exact_histogram_cuda(
    at::Tensor grad_flat,
    int64_t period,
    at::Tensor bin_counts
);

at::Tensor average_within_block_variance_cuda(
    at::Tensor values,
    int64_t period,
    bool input_is_squared
);

at::Tensor average_within_block_coefficient_of_variation_cuda(
    at::Tensor values,
    int64_t period,
    bool input_is_squared
);

void gefen_lloyd_accumulate_cuda(
    at::Tensor grad_flat,
    at::Tensor codebook,
    int64_t period,
    at::Tensor sums,
    at::Tensor counts
);

void gefen_lloyd_mse_cuda(
    at::Tensor grad_flat,
    at::Tensor old_codebook,
    at::Tensor new_codebook,
    int64_t period,
    at::Tensor mse_normalized_sum,
    at::Tensor mse_original_sum
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "automatic_gefen_fused_update_cuda",
        &automatic_gefen_fused_update_cuda,
        "Fused automatic Gefen momentum/state/parameter update (CUDA)",
        py::arg("p"),
        py::arg("grad_view"),
        py::arg("m_sign"),
        py::arg("m_magnitude"),
        py::arg("stepsize"),
        py::arg("codebook"),
        py::arg("packed_indices"),
        py::arg("beta1"),
        py::arg("lr")
    );
    m.def(
        "automatic_gefen_fused_update_from_vmean_cuda",
        &automatic_gefen_fused_update_from_vmean_cuda,
        "Fused automatic Gefen update with in-kernel Adam stepsize (CUDA)",
        py::arg("p"),
        py::arg("grad_view"),
        py::arg("m_sign"),
        py::arg("m_magnitude"),
        py::arg("vmean"),
        py::arg("codebook"),
        py::arg("packed_indices"),
        py::arg("beta1"),
        py::arg("lr"),
        py::arg("bias_correction_1"),
        py::arg("bias_correction_2"),
        py::arg("eps")
    );
    m.def(
        "automatic_gefen_period1_update_cuda",
        &automatic_gefen_period1_update_cuda,
        "Elementwise fused Gefen update for automatic period 1 (CUDA)",
        py::arg("p"),
        py::arg("grad_flat"),
        py::arg("m_sign"),
        py::arg("m_magnitude"),
        py::arg("vmean"),
        py::arg("codebook"),
        py::arg("period_one_divisor"),
        py::arg("beta1"),
        py::arg("beta2"),
        py::arg("lr"),
        py::arg("bias_correction_1"),
        py::arg("bias_correction_2"),
        py::arg("eps")
    );
    m.def(
        "automatic_vmean_update_cuda",
        &automatic_vmean_update_cuda,
        "Fused automatic vmean update (CUDA)",
        py::arg("vmean"),
        py::arg("grad_view"),
        py::arg("beta2")
    );
    m.def(
        "gefen_exact_histogram_cuda",
        &gefen_exact_histogram_cuda,
        "Accumulate exact-DP histogram counts from raw gradients on CUDA"
    );
    m.def(
        "average_within_block_variance_cuda",
        &average_within_block_variance_cuda,
        "Average within-block variance for one candidate period (CUDA)",
        py::arg("values"),
        py::arg("period"),
        py::arg("input_is_squared")
    );
    m.def(
        "average_within_block_coefficient_of_variation_cuda",
        &average_within_block_coefficient_of_variation_cuda,
        "Average within-block coefficient of variation for one candidate period (CUDA)",
        py::arg("values"),
        py::arg("period"),
        py::arg("input_is_squared")
    );
    m.def(
        "gefen_lloyd_accumulate_cuda",
        &gefen_lloyd_accumulate_cuda,
        "Accumulate Lloyd-Max codebook sums/counts from raw gradients on CUDA"
    );
    m.def(
        "gefen_lloyd_mse_cuda",
        &gefen_lloyd_mse_cuda,
        "Accumulate Lloyd-Max logging MSE terms from raw gradients on CUDA"
    );
}
