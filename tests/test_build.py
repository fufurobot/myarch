"""Tests for the build orchestrator.

Written BEFORE build.py exists (TDD red phase). `docker` is never actually
invoked: the docker call is captured so the tests stay hermetic and fast.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from myarch import build, detect


class FakeDocker:
    """Records docker invocations instead of running them."""

    def __init__(self, returncode: int = 0, stdout: str = "Successfully built abc123\n"):
        self.calls = []
        self.returncode = returncode
        self.stdout = stdout

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, self.returncode, stdout=self.stdout, stderr="")

    @property
    def last(self):
        assert self.calls, "docker was never invoked"
        return self.calls[-1]


@pytest.fixture
def gguf(tmp_path) -> Path:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 32)
    return model


@pytest.fixture
def hf_dir(tmp_path) -> Path:
    folder = tmp_path / "hf-model"
    folder.mkdir()
    (folder / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"\x00" * 16)
    return folder


@pytest.fixture
def cpu_env(monkeypatch):
    """Force a CPU-only host so tests are deterministic on any machine."""
    monkeypatch.setattr(
        build.detect, "detect_gpu", lambda: detect.GpuInfo("none", False, False, "", None)
    )


class TestContextBuilding:
    def test_gguf_on_cpu_selects_llamacpp(self, gguf, cpu_env):
        ctx = build.build_context(str(gguf))
        assert ctx["llm_backend"] == "llama.cpp"
        assert ctx["torch_backend"] == "cpu"
        assert ctx["model_kind"] == "gguf"
        assert ctx["gpu_vendor"] == "none"

    def test_hf_dir_on_cpu_falls_back_to_llamacpp(self, hf_dir, cpu_env):
        """A HuggingFace folder cannot run vLLM without a GPU."""
        ctx = build.build_context(str(hf_dir))
        assert ctx["model_kind"] == "huggingface"
        assert ctx["llm_backend"] == "llama.cpp"

    def test_nvidia_host_selects_vllm(self, hf_dir, monkeypatch):
        monkeypatch.setattr(
            build.detect,
            "detect_gpu",
            lambda: detect.GpuInfo("nvidia", True, True, "RTX 4090", "550.54.14"),
        )
        ctx = build.build_context(str(hf_dir))
        assert ctx["llm_backend"] == "vllm"
        assert ctx["torch_backend"] == "cu128"
        assert ctx["gpu_driver_package"] == "nvidia-open-dkms"

    def test_gguf_on_nvidia_host_still_uses_llamacpp(self, gguf, monkeypatch):
        """vLLM cannot read gguf, so the model format wins over the GPU."""
        monkeypatch.setattr(
            build.detect,
            "detect_gpu",
            lambda: detect.GpuInfo("nvidia", True, True, "RTX 4090", "550.54.14"),
        )
        ctx = build.build_context(str(gguf))
        assert ctx["llm_backend"] == "llama.cpp"
        assert ctx["torch_backend"] == "cu128"

    def test_remote_repo_id(self, cpu_env):
        ctx = build.build_context("meta-llama/Llama-3-8B")
        assert ctx["model_repo_id"] == "meta-llama/Llama-3-8B"
        assert ctx["model_kind"] == "huggingface"

    def test_llm_port_matches_backend(self, gguf, hf_dir, cpu_env):
        assert build.build_context(str(gguf))["llm_port"] == 8080
        assert build.build_context("meta-llama/Llama-3-8B")["llm_port"] == 8080

    def test_code_server_port_never_collides(self, gguf, cpu_env):
        ctx = build.build_context(str(gguf))
        assert ctx["code_server_port"] != ctx["llm_port"]

    def test_python_version_is_propagated(self, gguf, cpu_env):
        ctx = build.build_context(str(gguf), python_script=None)
        assert ctx["python_version"] == detect.python_version_for_script(None)


class TestRenderToFile:
    def test_writes_dockerfile(self, gguf, cpu_env, tmp_path):
        out = tmp_path / "Dockerfile"
        build.render_dockerfile(build.build_context(str(gguf)), out)
        text = out.read_text(encoding="utf-8")
        assert "FROM archlinux" in text
        assert "{{" not in text

    def test_rendered_context_is_self_consistent(self, gguf, cpu_env, tmp_path):
        out = tmp_path / "Dockerfile"
        ctx = build.build_context(str(gguf))
        build.render_dockerfile(ctx, out)
        text = out.read_text(encoding="utf-8")
        if ctx["llm_backend"] == "llama.cpp":
            assert "llama-server" in text
            assert "uv pip install --python /opt/venv/bin/python vllm" not in text


class TestBuildArguments:
    def test_default_tag_is_derived(self, gguf, cpu_env):
        args = build.parse_args(["--model", str(gguf)])
        assert args.tag

    def test_dry_run_does_not_call_docker(self, gguf, cpu_env, tmp_path, monkeypatch, capsys):
        docker = FakeDocker()
        monkeypatch.setattr(build, "_run", docker)
        rc = build.main(
            ["--model", str(gguf), "--dry-run", "--output", str(tmp_path / "Dockerfile")]
        )
        assert rc == 0
        assert docker.calls == [], "dry run must not invoke docker"
        assert "FROM archlinux" in capsys.readouterr().out

    def test_build_invokes_docker_build(self, gguf, cpu_env, tmp_path, monkeypatch):
        docker = FakeDocker()
        monkeypatch.setattr(build, "_run", docker)
        rc = build.main(
            [
                "--model", str(gguf),
                "--tag", "myarch:test",
                "--output", str(tmp_path / "Dockerfile"),
            ]
        )
        assert rc == 0
        cmd = docker.last
        assert cmd[0] == "docker"
        assert cmd[1] == "build"
        assert "myarch:test" in cmd

    def test_gpu_flag_added_when_gpu_usable(self, gguf, tmp_path, monkeypatch):
        monkeypatch.setattr(
            build.detect,
            "detect_gpu",
            lambda: detect.GpuInfo("nvidia", True, True, "RTX 4090", "550.54.14"),
        )
        docker = FakeDocker()
        monkeypatch.setattr(build, "_run", docker)
        build.main(["--model", str(gguf), "--output", str(tmp_path / "Dockerfile"), "--gpus"])
        assert "--gpus" in docker.last

    def test_no_gpu_flag_on_cpu_host(self, gguf, cpu_env, tmp_path, monkeypatch):
        docker = FakeDocker()
        monkeypatch.setattr(build, "_run", docker)
        build.main(["--model", str(gguf), "--output", str(tmp_path / "Dockerfile")])
        assert "--gpus" not in docker.last

    def test_multiple_tags_supported(self, gguf, cpu_env, tmp_path, monkeypatch):
        docker = FakeDocker()
        monkeypatch.setattr(build, "_run", docker)
        build.main(
            [
                "--model", str(gguf),
                "--output", str(tmp_path / "Dockerfile"),
                "--tag", "myarch:1",
                "--extra-tag", "myarch:latest",
            ]
        )
        cmd = docker.last
        assert cmd.count("-t") == 2

    def test_no_cache_flag(self, gguf, cpu_env, tmp_path, monkeypatch):
        docker = FakeDocker()
        monkeypatch.setattr(build, "_run", docker)
        build.main(["--model", str(gguf), "--output", str(tmp_path / "Dockerfile"), "--no-cache"])
        assert "--no-cache" in docker.last

    def test_failed_docker_build_returns_nonzero(self, gguf, cpu_env, tmp_path, monkeypatch):
        monkeypatch.setattr(build, "_run", FakeDocker(returncode=1))
        rc = build.main(["--model", str(gguf), "--output", str(tmp_path / "Dockerfile")])
        assert rc != 0

    def test_missing_model_is_a_clean_error(self, cpu_env, capsys):
        rc = build.main(["--model", "/nonexistent/model.gguf"])
        assert rc == 2
        assert "model" in capsys.readouterr().err.lower()


class TestModelStaging:
    def test_gguf_is_staged_into_build_context(self, gguf, cpu_env, tmp_path):
        ctx = build.build_context(str(gguf))
        staged = build.stage_model(ctx, context_dir=tmp_path)
        assert staged["model_source"] == "models/model.gguf"
        assert (tmp_path / "models" / "model.gguf").is_file()

    def test_hf_dir_is_staged_into_build_context(self, hf_dir, cpu_env, tmp_path):
        ctx = build.build_context(str(hf_dir))
        staged = build.stage_model(ctx, context_dir=tmp_path)
        assert staged["model_source"] == "models/hf_model"
        assert (tmp_path / "models" / "hf_model" / "config.json").is_file()

    def test_remote_repo_needs_no_staging(self, cpu_env, tmp_path):
        ctx = build.build_context("meta-llama/Llama-3-8B")
        staged = build.stage_model(ctx, context_dir=tmp_path)
        assert not (tmp_path / "models").exists()
        assert staged["model_repo_id"] == "meta-llama/Llama-3-8B"

    def test_staged_context_is_copied_not_moved(self, gguf, cpu_env, tmp_path):
        """Staging must not destroy the user's original model file."""
        ctx = build.build_context(str(gguf))
        build.stage_model(ctx, context_dir=tmp_path)
        assert gguf.is_file()

    def test_large_gguf_is_hardlinked_when_possible(self, gguf, cpu_env, tmp_path, monkeypatch):
        """Hardlinking avoids duplicating multi-GB weights."""
        linked = {"count": 0}

        def fake_link(src, dst):
            linked["count"] += 1
            return Path(dst).write_bytes(Path(src).read_bytes())

        monkeypatch.setattr(build.os, "link", fake_link, raising=False)
        ctx = build.build_context(str(gguf))
        build.stage_model(ctx, context_dir=tmp_path)
        assert linked["count"] == 1


class TestDockerfileDetection:
    def test_dockerfile_is_ignored_by_git(self):
        """The rendered Dockerfile is a build artifact and must not be committed."""
        gitignore = Path(".gitignore").read_text(encoding="utf-8")
        assert "Dockerfile" in gitignore
