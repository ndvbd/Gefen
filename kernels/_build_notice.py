from importlib.machinery import EXTENSION_SUFFIXES
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from typing import Optional

import torch
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import CUDA_HOME, get_default_build_root, load

BUILD_NOTICE = "Building Gefen kernels since it is the first run. Please wait..."


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _sanitize_cache_part(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _run_command(args) -> str:
    try:
        return subprocess.check_output(
            args, stderr=subprocess.STDOUT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Failed to run {} while preparing Gefen CUDA kernels: {}".format(
                args[0], exc
            )
        ) from exc


def _nvcc_path() -> str:
    if CUDA_HOME is not None:
        candidate = Path(CUDA_HOME) / "bin" / "nvcc"
        if candidate.exists():
            return str(candidate)

    candidate = shutil.which("nvcc")
    if candidate is not None:
        return candidate

    raise RuntimeError(
        "Gefen fused CUDA kernels require nvcc, but it was not found. "
        "Install a CUDA toolkit compatible with your PyTorch CUDA build and set CUDA_HOME if needed.\n"
        "{}\n\n{}".format(_cuda_diagnostic(None, None), _cuda_install_instructions())
    )


def _candidate_cuda_roots(nvcc: Optional[str]) -> list[Path]:
    roots = []
    for raw_root in (
        CUDA_HOME,
        os.environ.get("CUDA_HOME"),
        os.environ.get("CONDA_PREFIX"),
    ):
        if raw_root:
            roots.append(Path(raw_root))
    if nvcc is not None:
        roots.append(Path(nvcc).resolve().parents[1])

    unique_roots = []
    seen = set()
    for root in roots:
        resolved = str(root)
        if resolved not in seen:
            unique_roots.append(root)
            seen.add(resolved)
    return unique_roots


def _cuda_runtime_header_candidates(nvcc: Optional[str]) -> list[Path]:
    candidates = []
    for root in _candidate_cuda_roots(nvcc):
        candidates.append(root / "include" / "cuda_runtime.h")
        candidates.append(
            root / "targets" / "x86_64-linux" / "include" / "cuda_runtime.h"
        )
    return candidates


def _find_cuda_runtime_header(nvcc: Optional[str]) -> Optional[Path]:
    for candidate in _cuda_runtime_header_candidates(nvcc):
        if candidate.exists():
            return candidate
    return None


def _parse_cuda_release(output: str):
    match = re.search(r"release\s+([0-9]+)\.([0-9]+)", output)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _torch_cuda_version():
    if torch.version.cuda is None:
        return None

    parts = torch.version.cuda.split(".")
    if len(parts) < 2:
        return None
    return int(parts[0]), int(parts[1])


def _infer_torch_cuda_arch_list() -> Optional[str]:
    if not torch.cuda.is_available():
        return None

    device_capability = torch.cuda.get_device_capability()
    supported_capabilities = []
    for arch in torch.cuda.get_arch_list():
        if arch.startswith("sm_"):
            digits = re.findall(r"\d+", arch)
            if digits:
                sm = int(digits[0])
                supported_capabilities.append((sm // 10, sm % 10))

    if supported_capabilities:
        major, minor = min(device_capability, max(supported_capabilities))
        return "{}.{}+PTX".format(major, minor)

    major, minor = device_capability
    return "{}.{}".format(major, minor)


def _current_conda_prefix() -> Optional[str]:
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        return prefix
    return None


def _current_host_compiler() -> Optional[str]:
    for env_name in ("CC", "CXX"):
        candidate = os.environ.get(env_name)
        if candidate:
            return candidate

    for compiler_name in (
        "x86_64-conda-linux-gnu-cc",
        "gcc",
        "cc",
    ):
        candidate = shutil.which(compiler_name)
        if candidate is not None:
            return candidate
    return None


def _compiler_version_diagnostic() -> Optional[str]:
    compiler = _current_host_compiler()
    if compiler is None:
        return None

    try:
        version_output = _run_command([compiler, "--version"])
    except RuntimeError:
        return "  host_compiler: {} (failed to query --version)".format(compiler)
    first_line = version_output.splitlines()[0] if version_output else "<empty output>"
    return "  host_compiler: {} | {}".format(compiler, first_line)


def _cuda_diagnostic(nvcc: Optional[str], nvcc_output: Optional[str]) -> str:
    lines = [
        "Gefen CUDA build diagnostic:",
        "  python: {}".format(sys.executable),
        "  python_version: {}.{}.{}".format(
            sys.version_info.major, sys.version_info.minor, sys.version_info.micro
        ),
        "  torch: {}".format(torch.__version__),
        "  torch.version.cuda: {}".format(torch.version.cuda),
        "  torch.cuda.is_available: {}".format(torch.cuda.is_available()),
        "  CONDA_PREFIX: {}".format(_current_conda_prefix()),
        "  CUDA_HOME: {}".format(CUDA_HOME),
        "  nvcc: {}".format(nvcc),
    ]
    if nvcc_output is not None:
        lines.append("  nvcc --version: {}".format(nvcc_output.replace("\n", " | ")))
    compiler_line = _compiler_version_diagnostic()
    if compiler_line is not None:
        lines.append(compiler_line)
    cuda_runtime_header = _find_cuda_runtime_header(nvcc)
    lines.append("  cuda_runtime.h: {}".format(cuda_runtime_header))
    return "\n".join(lines)


def _cuda_install_instructions() -> str:
    torch_cuda = _torch_cuda_version()
    if torch_cuda is None:
        return (
            "Gefen fused CUDA kernels need a CUDA-enabled PyTorch build. Install a CUDA PyTorch wheel/conda package "
            "first, then relaunch Python and try again."
        )

    major, minor = torch_cuda
    conda_label = "cuda-{}.{}.0".format(major, minor)
    conda_cuda = "{}.{}".format(major, minor)
    inferred_arch_list = _infer_torch_cuda_arch_list()
    if inferred_arch_list is None:
        torch_cuda_arch_list_command = (
            "# Could not infer TORCH_CUDA_ARCH_LIST because torch.cuda.is_available() is False.\n"
            "    # Run on a machine where CUDA is visible, or set TORCH_CUDA_ARCH_LIST manually."
        )
    else:
        torch_cuda_arch_list_command = 'export TORCH_CUDA_ARCH_LIST="{}"'.format(
            inferred_arch_list
        )
    conda_gcc = major

    return """Gefen needs a CUDA toolkit with nvcc and cuda_runtime.h that exactly matches torch.version.cuda.
The recommended and tested setup is to install the CUDA build environment with conda:

    conda install -c nvidia/label/{conda_label} -c conda-forge cuda-nvcc={conda_cuda} cuda-cudart-dev={conda_cuda} gcc_linux-64={conda_gcc} gxx_linux-64={conda_gcc}
    export CUDA_HOME="$CONDA_PREFIX"
    export PATH="$CUDA_HOME/bin:$PATH"
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
    export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
    hash -r
    which nvcc
    nvcc --version
    "$CC" --version
    test -f "$CUDA_HOME/include/cuda_runtime.h" && echo "cuda_runtime.h found"
    test -f "$CUDA_HOME/targets/x86_64-linux/include/cuda_runtime.h" && echo "conda cuda_runtime.h found"
    conda list | grep -E 'cuda|nvcc|cudart'
    find "$CONDA_PREFIX" -name cuda_runtime.h
    {torch_cuda_arch_list_command}
    GEFEN_FORCE_REBUILD=1 GEFEN_VERBOSE_BUILD=1 python your_training_script.py

If you do not use conda, install CUDA Toolkit {conda_cuda} by any method supported by your operating system or package manager, then set CUDA_HOME to that toolkit before rebuilding.
""".format(
        major=major,
        minor=minor,
        conda_label=conda_label,
        conda_cuda=conda_cuda,
        conda_gcc=conda_gcc,
        torch_cuda_arch_list_command=torch_cuda_arch_list_command,
    )


def _check_cuda_toolkit() -> tuple[str, str]:
    if torch.version.cuda is None:
        raise RuntimeError(
            "Gefen fused CUDA kernels require a CUDA-enabled PyTorch build, but torch.version.cuda is None.\n"
            "{}\n\n{}".format(
                _cuda_diagnostic(None, None), _cuda_install_instructions()
            )
        )

    nvcc = _nvcc_path()
    nvcc_output = _run_command([nvcc, "--version"])
    nvcc_version = _parse_cuda_release(nvcc_output)
    torch_cuda = _torch_cuda_version()
    if nvcc_version is None:
        raise RuntimeError(
            "Gefen could not parse the CUDA toolkit version from nvcc output.\n"
            "{}".format(_cuda_diagnostic(nvcc, nvcc_output))
        )
    if torch_cuda is None:
        raise RuntimeError(
            "Gefen could not parse torch.version.cuda='{}'.\n"
            "{}".format(torch.version.cuda, _cuda_diagnostic(nvcc, nvcc_output))
        )
    if nvcc_version != torch_cuda:
        raise RuntimeError(
            "Gefen CUDA toolkit version mismatch: nvcc reports {}.{}, "
            "but PyTorch was built with CUDA {}.{}.\n"
            "Install an nvcc toolkit that exactly matches torch.version.cuda and point CUDA_HOME to that toolkit.\n"
            "{}\n\n{}".format(
                nvcc_version[0],
                nvcc_version[1],
                torch_cuda[0],
                torch_cuda[1],
                _cuda_diagnostic(nvcc, nvcc_output),
                _cuda_install_instructions(),
            )
        )
    cuda_runtime_header = _find_cuda_runtime_header(nvcc)
    if cuda_runtime_header is None:
        candidates = "\n".join(
            "  {}".format(path) for path in _cuda_runtime_header_candidates(nvcc)
        )
        raise RuntimeError(
            "Gefen found nvcc, but the CUDA toolkit headers are missing. "
            "The JIT build needs cuda_runtime.h.\n"
            "Checked these locations:\n{}\n"
            "{}\n\n{}".format(
                candidates,
                _cuda_diagnostic(nvcc, nvcc_output),
                _cuda_install_instructions(),
            )
        )
    inferred_cuda_home = str(Path(nvcc).resolve().parents[1])
    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin" / "nvcc").exists():
        os.environ["CUDA_HOME"] = inferred_cuda_home
        cpp_extension.CUDA_HOME = inferred_cuda_home
    return nvcc, nvcc_output


def _build_dir(extension_name: str, kernel_dir: Path) -> Path:
    path_hash = hashlib.sha256(str(kernel_dir).encode("utf-8")).hexdigest()[:12]
    env_parts = [
        "py{}.{}".format(sys.version_info.major, sys.version_info.minor),
        "torch{}".format(torch.__version__),
        "cu{}".format(torch.version.cuda),
        "src{}".format(path_hash),
    ]
    env_key = "_".join(_sanitize_cache_part(part) for part in env_parts)
    return (
        Path(os.environ.get("GEFEN_KERNEL_BUILD_ROOT", get_default_build_root()))
        / "gefen"
        / env_key
        / extension_name
    )


def _cached_extension_exists(build_dir: Path, extension_name: str, sources) -> bool:

    for suffix in EXTENSION_SUFFIXES:
        extension_path = build_dir / "{}{}".format(extension_name, suffix)
        if extension_path.exists():
            extension_mtime = extension_path.stat().st_mtime
            for source in sources:
                if Path(source).stat().st_mtime > extension_mtime:
                    return False
            return True
    return False


def should_verbose_build(build_dir: Path, extension_name: str, sources) -> bool:
    if _cached_extension_exists(build_dir, extension_name, sources):
        return False

    print(BUILD_NOTICE)
    return True


def load_gefen_cuda_extension(extension_name: str, source_filenames):
    kernel_dir = Path(__file__).resolve().parent
    sources = [str(kernel_dir / filename) for filename in source_filenames]
    for source in sources:
        if not Path(source).exists():
            raise FileNotFoundError(
                "Gefen CUDA source file does not exist: {}".format(source)
            )

    build_dir = _build_dir(extension_name, kernel_dir)
    if _env_flag("GEFEN_FORCE_REBUILD") and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    will_build = not _cached_extension_exists(build_dir, extension_name, sources)
    nvcc = None
    nvcc_output = None
    if will_build:
        nvcc, nvcc_output = _check_cuda_toolkit()
    elif _env_flag("GEFEN_VERBOSE_BUILD"):
        print("Gefen found a cached CUDA extension; skipping CUDA toolkit preflight.")

    should_verbose_build(build_dir, extension_name, sources)
    verbose = will_build or _env_flag("GEFEN_VERBOSE_BUILD")
    if verbose:
        if nvcc is not None:
            print(_cuda_diagnostic(nvcc, nvcc_output))
        print("  Gefen extension build_dir: {}".format(build_dir))

    lock_path = build_dir / "lock"
    if lock_path.exists():
        print(
            "Gefen is waiting on PyTorch CUDA extension build lock:\n"
            "  {}\n"
            "If no nvcc/ninja process is active, this lock may be stale. "
            "Stop other Gefen runs and remove the lock or the extension build directory, then rerun.".format(
                lock_path
            ),
            flush=True,
        )

    try:
        return run_with_progress_dots(
            lambda: load(
                name=extension_name,
                sources=sources,
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                build_directory=str(build_dir),
                verbose=verbose,
            )
        )
    except Exception as exc:
        message = str(exc)
        if "unsupported GNU version" in message:
            raise RuntimeError(
                "Gefen CUDA JIT build failed because nvcc rejected the host compiler.\n"
                "Install a GCC/G++ 12 toolchain in the same environment as Python, export CC/CXX to that toolchain, "
                "and make sure nvcc exactly matches torch.version.cuda before rebuilding.\n"
                "{}\n\n{}".format(
                    _cuda_diagnostic(nvcc, nvcc_output), _cuda_install_instructions()
                )
            ) from exc
        if "Unknown CUDA arch" in message:
            raise RuntimeError(
                "Gefen CUDA JIT build failed because PyTorch rejected the CUDA architecture.\n"
                "Set TORCH_CUDA_ARCH_LIST using the commands below, then rebuild.\n"
                "{}\n\n{}".format(
                    _cuda_diagnostic(nvcc, nvcc_output), _cuda_install_instructions()
                )
            ) from exc
        if nvcc is None:
            nvcc, nvcc_output = _check_cuda_toolkit()
        raise RuntimeError(
            "Gefen CUDA JIT build failed.\n"
            "First verify that nvcc exactly matches torch.version.cuda and that CUDA_HOME points to the same environment "
            "as Python. Then rerun with GEFEN_FORCE_REBUILD=1 GEFEN_VERBOSE_BUILD=1 for a fresh rebuild.\n"
            "{}\n\n{}".format(
                _cuda_diagnostic(nvcc, nvcc_output), _cuda_install_instructions()
            )
        ) from exc


def run_with_progress_dots(fn):
    done = threading.Event()
    printed_dot = [False]

    def print_dots():
        while not done.wait(1.0):
            if not printed_dot[0]:
                print("Build CUDA kernel", end="", flush=True)
            print(".", end="", flush=True)
            printed_dot[0] = True

    thread = threading.Thread(target=print_dots, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        done.set()
        thread.join()
        if printed_dot[0]:
            print("", flush=True)
