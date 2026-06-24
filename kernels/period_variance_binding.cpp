#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

at::Tensor average_within_block_variance_cuda(
    at::Tensor values,
    int64_t period,
    bool input_is_squared
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "average_within_block_variance_cuda",
        &average_within_block_variance_cuda,
        "Average within-block variance for one candidate period (CUDA)",
        py::arg("values"),
        py::arg("period"),
        py::arg("input_is_squared")
    );
}
