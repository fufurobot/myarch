"""Host detection: GPU, Python version and model input resolution.

Every probe is defensive. A GPU that appears in ``lspci`` but whose driver is
not loaded is reported as ``present`` but **not** ``usable`` -- building a CUDA
image for such a host would produce an image that cannot start.

Requires Python 3.8+ (the project floor), so runtime type annotations use
``typing.Optional``/``Dict`` rather than PEP 604 syntax.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence

#: Oldest Python release that ``uv python install`` can still provide.
OLDEST_UV_PYTHON = "3.8"
#: Newest Python release we are willing to pin.
NEWEST_UV_PYTHON = "3.13"

#: NVIDIA driver major version at which Arch switched to the open kernel module.
NVIDIA_OPEN_MODULE_MIN_MAJOR = 515
#: Below this major version no Arch package exists and pikaur must be consulted.
NVIDIA_ARCH_SUPPORTED_MIN_MAJOR = 390

_MODEL_KINDS = ("gguf", "huggingface")
_HF_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


class GpuInfo(NamedTuple):
    """Result of probing one GPU vendor."""

    vendor: str  # "nvidia" | "amd" | "none"
    present: bool  # hardware visible to the OS
    usable: bool  # driver loaded and answering
    name: str  # marketing name when known
    driver_version: Optional[str]  # driver version when known

    @property
    def driver_package(self) -> str:
        if self.vendor == "nvidia":
            return nvidia_driver_package(self.driver_version)
        if self.vendor == "amd":
            return "xf86-video-amdgpu"
        return ""


class ModelSpec(NamedTuple):
    """How a user-supplied model should be packaged into the image."""

    kind: str  # "gguf" | "huggingface"
    host_path: str  # absolute host path, or the repo id for remote models
    repo_id: Optional[str]  # set for huggingface repo ids
    backend: str  # preferred backend: "llama.cpp" | "vllm"


def _path_exists(path: str) -> bool:
    """Indirection so tests can fake filesystem probes."""
    return os.path.exists(path)


def _run(cmd: Sequence[str], timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def nvidia_driver_package(driver_version: Optional[str]) -> str:
    """Pick the Arch package for an NVIDIA driver version.

    >= 515 uses the open kernel module; everything else (including an unknown
    version, where we cannot prove the open module is supported) uses the
    proprietary DKMS module, which is the safe default.
    """
    major = _major_version(driver_version)
    if major is None:
        return "nvidia-dkms"
    return "nvidia-open-dkms" if major >= NVIDIA_OPEN_MODULE_MIN_MAJOR else "nvidia-dkms"


def nvidia_needs_aur_fallback(driver_version: Optional[str]) -> bool:
    """True when the driver is too old for any official Arch package."""
    major = _major_version(driver_version)
    if major is None:
        return False
    return major < NVIDIA_ARCH_SUPPORTED_MIN_MAJOR


def _major_version(driver_version: Optional[str]) -> Optional[int]:
    if not driver_version:
        return None
    match = re.match(r"\s*(\d+)", str(driver_version))
    if not match:
        return None
    return int(match.group(1))


def detect_nvidia() -> GpuInfo:
    """Probe for a usable NVIDIA GPU.

    ``nvidia-smi`` answering is the only proof that the driver works. Merely
    finding the card in ``lspci`` (or the binary on disk) is not enough.
    """
    if not shutil.which("nvidia-smi"):
        return GpuInfo("nvidia", _lspci_has("nvidia"), False, "", None)
    try:
        proc = _run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            timeout=15,
        )
    except Exception:
        return GpuInfo("nvidia", True, False, "", None)

    if proc.returncode != 0 or not (proc.stdout or "").strip():
        # Present in the machine but the driver is not loaded.
        return GpuInfo("nvidia", True, False, "", None)

    line = _first_line(proc.stdout)
    parts = [p.strip() for p in line.split(",")]
    name = parts[0] if parts else ""
    driver = parts[1] if len(parts) > 1 else None
    return GpuInfo("nvidia", True, True, name, driver)


def detect_amd() -> GpuInfo:
    """Probe for an AMD GPU.

    ``/dev/kfd`` proves the ROCm kernel driver is live; otherwise fall back to
    ``lspci`` identification, which only proves presence.
    """
    if _path_exists("/dev/kfd"):
        return GpuInfo("amd", True, True, "AMD GPU (kfd)", None)

    # No /dev/kfd, but an identified AMD display adapter is still driven by the
    # in-kernel amdgpu module on any modern Arch host, which is what ROCm needs.
    present = _lspci_has("amd") or _lspci_has("advanced micro devices")
    if present:
        return GpuInfo("amd", True, True, "AMD GPU", None)
    return GpuInfo("amd", False, False, "", None)


def _lspci_has(needle: str) -> bool:
    if not shutil.which("lspci"):
        return False
    try:
        proc = _run(["lspci", "-nn"], timeout=15)
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    needle = needle.lower()
    for line in (proc.stdout or "").splitlines():
        low = line.lower()
        if "vga" not in low and "3d controller" not in low and "display" not in low:
            continue
        if needle in low:
            return True
    return False


def detect_gpu() -> GpuInfo:
    """Combine vendor probes, preferring a usable NVIDIA GPU."""
    nvidia = detect_nvidia()
    if nvidia.usable:
        return nvidia
    amd = detect_amd()
    if amd.usable:
        return amd
    # Nothing usable: report whichever card is physically present so the build
    # can tell "no GPU" apart from "GPU without a driver".
    if nvidia.present:
        return nvidia
    if amd.present:
        return amd
    return GpuInfo("none", False, False, "", None)


def torch_backend_for(vendor: str, driver_version: Optional[str]) -> str:
    """Map a GPU vendor to a PyTorch wheel index flavour."""
    if vendor == "nvidia":
        return "cu128"
    if vendor == "amd":
        return "rocm6.4"
    return "cpu"


def python_version_for_script(script_path: Optional[str]) -> str:
    """Resolve the Python version to install for the user's script.

    Order of evidence: ``pyproject.toml`` next to the script, ``.python-version``,
    the script's shebang, then the running interpreter. Anything older than the
    oldest release uv supports is clamped up to :data:`OLDEST_UV_PYTHON`.
    """
    candidate: Optional[str] = None

    if script_path:
        target = Path(script_path)
        if target.is_file():
            # The script's own directory holds the project metadata.
            base = target.parent
            candidate = _version_from_shebang(target)
        else:
            # A directory, or a not-yet-created script path: use its parent.
            base = target if target.is_dir() else target.parent
        candidate = candidate or _version_from_pyproject(base) or _version_from_python_version_file(base)

    if candidate is None:
        candidate = "%d.%d" % (sys.version_info[0], sys.version_info[1])

    return _clamp_python(candidate)


def _version_from_shebang(path: Path) -> Optional[str]:
    try:
        first = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if not first:
        return None
    match = re.search(r"python(\d+\.\d+)", first[0])
    if match:
        return match.group(1)
    # python2 style shebangs are intentionally caught later by the clamp.
    if re.search(r"python2(\.\d+)?\b", first[0]):
        return "2.7"
    return None


def _version_from_pyproject(base: Path) -> Optional[str]:
    pyproject = base / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        text = pyproject.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r'requires-python\s*=\s*["\']>=?\s*(\d+\.\d+)', text)
    if match:
        return match.group(1)
    return None


def _version_from_python_version_file(base: Path) -> Optional[str]:
    pin = base / ".python-version"
    if not pin.is_file():
        return None
    try:
        text = pin.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    match = re.match(r"(\d+\.\d+)", text)
    if match:
        return match.group(1)
    return None


def _clamp_python(version: str) -> str:
    major, minor = _parse_version(version)
    if major is None:
        return OLDEST_UV_PYTHON
    if major < 3:
        return OLDEST_UV_PYTHON
    lo_major, lo_minor = _parse_version(OLDEST_UV_PYTHON)
    hi_major, hi_minor = _parse_version(NEWEST_UV_PYTHON)
    if (major, minor) < (lo_major, lo_minor):  # type: ignore[operator]
        return OLDEST_UV_PYTHON
    if (major, minor) > (hi_major, hi_minor):  # type: ignore[operator]
        return NEWEST_UV_PYTHON
    return "%d.%d" % (major, minor)


def _parse_version(version: Optional[str]):
    if not version:
        return None, None
    match = re.match(r"\s*(\d+)\.(\d+)", str(version))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def resolve_model(model_input: str) -> ModelSpec:
    """Classify a user-supplied model as a gguf file, local folder or repo id."""
    path = Path(model_input).expanduser()

    if path.exists():
        resolved = str(path.resolve())
        if path.is_dir():
            # A HuggingFace folder normally carries config.json. We do not hard
            # fail without it: quantised or partially downloaded snapshots are
            # still valid inputs, and the container reports the problem clearly.
            return ModelSpec("huggingface", resolved, None, "vllm")
        if path.suffix.lower() != ".gguf":
            raise ValueError(
                "%s is not a .gguf file; pass a .gguf file or a HuggingFace folder" % resolved
            )
        return ModelSpec("gguf", resolved, None, "llama.cpp")

    if path.suffix.lower() == ".gguf":
        raise FileNotFoundError("no such gguf file: %s" % model_input)

    if _HF_REPO_RE.match(model_input.strip()):
        repo_id = model_input.strip()
        return ModelSpec("huggingface", repo_id, repo_id, "vllm")

    raise FileNotFoundError("cannot resolve model input: %s" % model_input)


def is_remote_model(spec: ModelSpec) -> bool:
    """True when the model is a HuggingFace repo id rather than a host path."""
    return spec.repo_id is not None


def choose_llm_backend(gpu_vendor: str, spec: ModelSpec) -> str:
    """Pick the serving backend.

    vLLM only runs on a GPU and only reads HuggingFace-format weights, so a
    CPU-only host or a ``.gguf`` model always falls back to llama.cpp.
    """
    if spec.kind == "gguf":
        return "llama.cpp"
    if gpu_vendor in ("nvidia", "amd"):
        return "vllm"
    return "llama.cpp"


def detect_host(script_path: Optional[str] = None) -> dict:
    """Collect everything the Dockerfile template needs."""
    gpu = detect_gpu()
    return {
        "gpu_vendor": gpu.vendor if gpu.usable else "none",
        "gpu_present": gpu.present,
        "gpu_usable": gpu.usable,
        "gpu_name": gpu.name,
        "gpu_driver_version": gpu.driver_version or "",
        "gpu_driver_package": gpu.driver_package if gpu.usable else "",
        "gpu_needs_aur_fallback": nvidia_needs_aur_fallback(gpu.driver_version) if gpu.vendor == "nvidia" else False,
        "python_version": python_version_for_script(script_path),
        "torch_backend": torch_backend_for(gpu.vendor if gpu.usable else "none", gpu.driver_version),
    }
