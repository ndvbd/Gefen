from ._build_notice import load_gefen_cuda_extension

_EXTENSION_MODULE = None


def load_extension():
    global _EXTENSION_MODULE
    if _EXTENSION_MODULE is not None:
        return _EXTENSION_MODULE

    _EXTENSION_MODULE = load_gefen_cuda_extension(
        "gefen_cuda_ext",
        [
            "gefen_cuda_binding.cpp",
            "automatic_gefen_fused_kernel.cu",
            "automatic_vmean_kernel.cu",
            "exact_histogram_fused_kernel.cu",
            "period_variance_kernel.cu",
            "lloyd_max_fused_kernel.cu",
        ],
    )
    return _EXTENSION_MODULE
