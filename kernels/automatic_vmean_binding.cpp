#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

void automatic_vmean_update_cuda(
    at::Tensor vmean,
    at::Tensor grad_view,
    double beta2
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "automatic_vmean_update_cuda",
        &automatic_vmean_update_cuda,
        "Fused automatic vmean update (CUDA)",
        py::arg("vmean"),
        py::arg("grad_view"),
        py::arg("beta2")
    );
}
