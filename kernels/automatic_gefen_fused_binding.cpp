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
}
