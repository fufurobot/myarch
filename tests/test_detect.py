"""Tests for host hardware/GPU detection.

Written BEFORE the implementation exists (TDD red phase).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from myarch import detect


class TestNvidia:
    def test_detects_working_nvidia_gpu(self, monkeypatch):
        """A responding nvidia-smi means the driver is actually usable."""

        def fake_run(cmd, **kwargs):
            assert cmd[0] == "nvidia-smi"
            return subprocess.CompletedProcess(
                cmd, 0, stdout="NVIDIA GeForce RTX 4090, 550.54.14\n", stderr=""
            )

        monkeypatch.setattr(detect.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        monkeypatch.setattr(detect.subprocess, "run", fake_run)

        result = detect.detect_nvidia()
        assert result.usable is True
        assert result.vendor == "nvidia"
        assert "4090" in result.name
        assert result.driver_version == "550.54.14"

    def test_lspci_presence_without_working_driver_is_not_usable(self, monkeypatch):
        """Regression: this host has a GM107M in lspci but nvidia-smi fails."""
        monkeypatch.setattr(detect.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 9, stdout="", stderr="NVIDIA-SMI has failed")

        monkeypatch.setattr(detect.subprocess, "run", fake_run)
        result = detect.detect_nvidia()
        assert result.usable is False
        assert result.present is True  # card is there, driver is not loaded

    def test_no_nvidia_binary_at_all(self, monkeypatch):
        monkeypatch.setattr(detect.shutil, "which", lambda name: None)
        result = detect.detect_nvidia()
        assert result.usable is False
        assert result.vendor == "nvidia"

    @pytest.mark.parametrize(
        "driver,expected_pkg",
        [
            ("550.54.14", "nvidia-open-dkms"),
            ("515.0.0", "nvidia-open-dkms"),
            ("470.239.06", "nvidia-dkms"),
            ("390.157", "nvidia-dkms"),
        ],
    )
    def test_driver_package_selection(self, driver, expected_pkg):
        """Newer drivers get the open kernel module, old ones the proprietary DKMS."""
        assert detect.nvidia_driver_package(driver) == expected_pkg

    def test_old_driver_requires_aur_lookup(self):
        """Drivers below the Arch-supported floor must fall back to pikaur."""
        assert detect.nvidia_needs_aur_fallback("340.108") is True
        assert detect.nvidia_needs_aur_fallback("550.54.14") is False

    def test_malformed_driver_version_is_safe(self):
        assert detect.nvidia_driver_package("") == "nvidia-dkms"
        assert detect.nvidia_driver_package("unknown") == "nvidia-dkms"


class TestAmd:
    def test_detects_amdgpu_via_lspci(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 21\n",
                stderr="",
            )

        monkeypatch.setattr(detect.shutil, "which", lambda name: "/usr/bin/lspci")
        monkeypatch.setattr(detect, "_path_exists", lambda p: False)
        monkeypatch.setattr(detect.subprocess, "run", fake_run)

        result = detect.detect_amd()
        assert result.usable is True
        assert result.vendor == "amd"

    def test_no_amd_gpu(self, monkeypatch):
        monkeypatch.setattr(detect, "_path_exists", lambda p: False)

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 0, stdout="00:02.0 VGA compatible controller: Intel Corporation HD Graphics\n", stderr=""
            )

        monkeypatch.setattr(detect.shutil, "which", lambda name: "/usr/bin/lspci")
        monkeypatch.setattr(detect.subprocess, "run", fake_run)
        assert detect.detect_amd().usable is False


class TestGpuSummary:
    def test_prefers_nvidia_over_amd(self, monkeypatch):
        monkeypatch.setattr(
            detect, "detect_nvidia", lambda: detect.GpuInfo("nvidia", True, True, "RTX 4090", "550.1")
        )
        monkeypatch.setattr(
            detect, "detect_amd", lambda: detect.GpuInfo("amd", True, True, "Navi 21", None)
        )
        assert detect.detect_gpu().vendor == "nvidia"

    def test_cpu_only_when_nothing_usable(self, monkeypatch):
        monkeypatch.setattr(
            detect, "detect_nvidia", lambda: detect.GpuInfo("nvidia", False, False, "", None)
        )
        monkeypatch.setattr(detect, "detect_amd", lambda: detect.GpuInfo("amd", False, False, "", None))
        gpu = detect.detect_gpu()
        assert gpu.vendor == "none"
        assert gpu.usable is False

    def test_torch_backend_mapping(self):
        assert detect.torch_backend_for("nvidia", "550.54.14") == "cu128"
        assert detect.torch_backend_for("amd", None) == "rocm6.4"
        assert detect.torch_backend_for("none", None) == "cpu"


class TestPythonVersion:
    def test_reads_requires_python_from_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "x"\nrequires-python = ">=3.11"\n', encoding="utf-8"
        )
        assert detect.python_version_for_script(str(tmp_path / "app.py")) == "3.11"

    def test_reads_python_version_file(self, tmp_path):
        (tmp_path / ".python-version").write_text("3.10.4\n", encoding="utf-8")
        assert detect.python_version_for_script(str(tmp_path / "app.py")) == "3.10"

    def test_reads_shebang(self, tmp_path):
        script = tmp_path / "run.py"
        script.write_text("#!/usr/bin/env python3.9\nprint(1)\n", encoding="utf-8")
        assert detect.python_version_for_script(str(script)) == "3.9"

    def test_clamps_to_oldest_supported_version(self, tmp_path):
        """A 2.x script must clamp up to the oldest Python uv can install."""
        script = tmp_path / "legacy.py"
        script.write_text("#!/usr/bin/env python2.7\nprint 1\n", encoding="utf-8")
        assert detect.python_version_for_script(str(script)) == detect.OLDEST_UV_PYTHON

    def test_falls_back_to_host_python(self, monkeypatch):
        monkeypatch.setattr(detect.sys, "version_info", (3, 12, 1))
        assert detect.python_version_for_script(None) == "3.12"


class TestModelResolution:
    def test_gguf_file(self, tmp_path):
        model = tmp_path / "llama-3-8b.gguf"
        model.write_bytes(b"GGUF")
        result = detect.resolve_model(str(model))
        assert result.kind == "gguf"
        assert result.backend == "llama.cpp"
        assert Path(result.host_path) == model.resolve()

    def test_huggingface_directory(self, tmp_path):
        folder = tmp_path / "Qwen2.5-7B"
        folder.mkdir()
        (folder / "config.json").write_text("{}", encoding="utf-8")
        result = detect.resolve_model(str(folder))
        assert result.kind == "huggingface"
        assert result.backend == "vllm"

    def test_huggingface_repo_id(self):
        result = detect.resolve_model("meta-llama/Llama-3-8B")
        assert result.kind == "huggingface"
        assert result.backend == "vllm"
        assert result.repo_id == "meta-llama/Llama-3-8B"

    def test_gguf_on_cpu_forces_llama_cpp(self, tmp_path):
        """vLLM cannot read gguf, so a gguf model pins the backend to llama.cpp."""
        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF")
        assert detect.choose_llm_backend("none", detect.resolve_model(str(model))) == "llama.cpp"

    def test_gpu_with_hf_folder_chooses_vllm(self, tmp_path):
        folder = tmp_path / "hf"
        folder.mkdir()
        assert detect.choose_llm_backend("nvidia", detect.resolve_model(str(folder))) == "vllm"

    def test_missing_path_is_an_error(self):
        with pytest.raises(FileNotFoundError):
            detect.resolve_model("/nonexistent/path/model.gguf")


class TestModelConsistency:
    def test_cpu_with_hf_folder_is_rejected_or_downgraded(self, tmp_path):
        """No GPU + HF folder cannot run vLLM, so backend must fall back to llama.cpp."""
        folder = tmp_path / "hf"
        folder.mkdir()
        assert detect.choose_llm_backend("none", detect.resolve_model(str(folder))) == "llama.cpp"
