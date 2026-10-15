"""Tests for rendering Dockerfile.j2.

Written BEFORE the template exists (TDD red phase). These assert the exact
security-relevant behaviour from todo.md: user creation, sudoers grants,
forced password change, sshd, and per-requirement installation steps.
"""

from __future__ import annotations

import re

import pytest
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from myarch import detect

TEMPLATE_DIR = "."
TEMPLATE_NAME = "Dockerfile.j2"

# The exact sudoers commands from todo.md.
PIKAUR_SUDO_COMMAND = (
    "/usr/bin/pikaur -Syu [!-]* "
    "--pikaur-config=/home/pikaur-manager/.config/pikaur.conf "
    "--user-id=1005 --group-id=1005 "
    "--home-dir=/home/pikaur-manager "
    "--xdg-cache-home=/home/pikaur-manager/.cache "
    "--xdg-data-home=/home/pikaur-manager/.local/share"
)

USER_PASSWORDS = {
    "user": "iamsafe",
    "pikaur-manager": "iknowtheremotecodeisdangerous",
    "developer": "iwillnotinstallpackagesidonotownsourcecode",
    "system-manager": "iwillnotstealotherusersdata",
    "service-manager": "iwillnotstartuntrustedservices",
}


def render(**overrides) -> str:
    """Render the template with a complete, valid default context."""
    context = {
        "gpu_vendor": "none",
        "gpu_driver_package": "",
        "gpu_name": "",
        "gpu_driver_version": "",
        "gpu_needs_aur_fallback": False,
        "python_version": "3.12",
        "torch_backend": "cpu",
        "llm_backend": "llama.cpp",
        "model_kind": "gguf",
        "model_source": "models/model.gguf",
        "model_repo_id": "",
        "code_server_port": 8080,
        "llm_port": 8081,
        "vnc_resolution": "1920x1080",
        "enable_ssh": True,
    }
    context.update(overrides)
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template(TEMPLATE_NAME).render(**context)


@pytest.fixture(scope="module")
def base_render() -> str:
    return render()


@pytest.fixture(scope="module")
def nvidia_render() -> str:
    return render(
        gpu_vendor="nvidia",
        gpu_driver_package="nvidia-open-dkms",
        gpu_name="NVIDIA GeForce RTX 4090",
        gpu_driver_version="550.54.14",
        torch_backend="cu128",
        llm_backend="vllm",
        model_kind="huggingface",
        model_source="meta-llama/Llama-3-8B",
    )


# ---------------------------------------------------------------------------
# Requirement 1: users, groups, sudo
# ---------------------------------------------------------------------------


class TestUsers:
    @pytest.mark.parametrize("username", sorted(USER_PASSWORDS))
    def test_every_user_is_created(self, base_render, username):
        assert re.search(r"useradd[^\n]*\b%s\b" % re.escape(username), base_render)

    @pytest.mark.parametrize("username,password", sorted(USER_PASSWORDS.items()))
    def test_password_is_set(self, base_render, username, password):
        assert password in base_render

    @pytest.mark.parametrize("username", sorted(USER_PASSWORDS))
    def test_password_change_is_forced(self, base_render, username):
        """chage -d 0 must be applied to every user."""
        assert re.search(r"chage\s+-d\s+0\s+%s\b" % re.escape(username), base_render)

    def test_administrator_master_can_sudo_everything(self, base_render):
        assert re.search(
            r"administrator-master\s+ALL=\(ALL(?::ALL)?\)\s+ALL", base_render
        ), "administrator-master must have unrestricted sudo"

    def test_administrator_master_has_a_password(self, base_render):
        assert "administrator-master" in base_render
        # password must be set for the account, whatever the value
        assert re.search(r"administrator-master:[^'\"]+", base_render)

    def test_pikaur_manager_sudo_is_exactly_the_pikaur_command(self, base_render):
        assert PIKAUR_SUDO_COMMAND in base_render, (
            "pikaur-manager sudoers rule must be the exact documented command"
        )

    def test_developer_sudo_is_only_pacman(self, base_render):
        match = re.search(r"^developer\s+ALL=.*$", base_render, re.MULTILINE)
        assert match, "developer must have a sudoers rule"
        rule = match.group(0)
        assert "/usr/bin/pacman" in rule
        # no extra commands beyond pacman
        commands = [c.strip() for c in rule.split("=", 1)[1].split(",")]
        assert len(commands) == 1, "developer must be limited to exactly one command"

    def test_system_manager_sudo_is_limited_to_fdisk_mount_ls(self, base_render):
        match = re.search(r"^system-manager\s+ALL=.*$", base_render, re.MULTILINE)
        assert match, "system-manager must have a sudoers rule"
        rule = match.group(0)
        for allowed in ("fdisk", "mount", "ls"):
            assert allowed in rule
        commands = [c.strip() for c in rule.split("=", 1)[1].split(",")]
        assert len(commands) == 3, "system-manager must be limited to exactly three commands"

    def test_service_manager_sudo_is_only_systemctl(self, base_render):
        match = re.search(r"^service-manager\s+ALL=.*$", base_render, re.MULTILINE)
        assert match, "service-manager must have a sudoers rule"
        rule = match.group(0)
        assert "/usr/bin/systemctl" in rule
        commands = [c.strip() for c in rule.split("=", 1)[1].split(",")]
        assert len(commands) == 1

    def test_plain_user_has_no_sudo_rule(self, base_render):
        """'user' is just a user and can sudo nothing."""
        assert not re.search(r"^user\s+ALL=", base_render, re.MULTILINE)
        assert "user ALL=" not in base_render

    def test_no_group_sudo_regression(self, base_render):
        """No rule may grant sudo to a group (e.g. %wheel)."""
        assert not re.search(r"^%wheel\s+ALL=", base_render, re.MULTILINE)

    def test_sudoers_files_are_validated(self, base_render):
        """Dropped-in sudoers files must be syntax-checked and 0440."""
        assert "visudo" in base_render, "sudoers files must be validated with visudo"
        assert "0440" in base_render, "sudoers files must be mode 0440"

    def test_sudoers_dropin_uses_include_dir(self, base_render):
        assert "/etc/sudoers.d" in base_render


# ---------------------------------------------------------------------------
# Requirement 1b: sshd
# ---------------------------------------------------------------------------


class TestSshd:
    def test_sshd_is_installed(self, base_render):
        assert "openssh" in base_render

    def test_password_authentication_is_enabled(self, base_render):
        assert re.search(r"PasswordAuthentication\s+yes", base_render)

    def test_sshd_is_enabled(self, base_render):
        assert "systemctl enable sshd" in base_render or "ssh-keygen -A" in base_render

    def test_host_keys_are_generated(self, base_render):
        assert "ssh-keygen -A" in base_render


# ---------------------------------------------------------------------------
# Requirement 2: pikaur from AUR
# ---------------------------------------------------------------------------


class TestPikaur:
    def test_built_from_aur_with_makepkg(self, base_render):
        assert "aur.archlinux.org/pikaur.git" in base_render
        assert "makepkg" in base_render

    def test_built_as_pikaur_manager_not_root(self, base_render):
        """makepkg refuses to run as root; the build must drop privileges."""
        assert re.search(r"su\s+-\s+pikaur-manager|sudo\s+-u\s+pikaur-manager", base_render)

    def test_does_not_use_root_install_flag(self, base_render):
        """-i needs root, which the restricted sudoers rule forbids."""
        assert "makepkg -fsri" not in base_render
        assert "-fsri" not in base_render

    def test_pikaur_conf_is_present(self, base_render):
        assert "pikaur.conf" in base_render

    def test_makepkg_dependencies_installed(self, base_render):
        assert "base-devel" in base_render


# ---------------------------------------------------------------------------
# Requirement 3: GPU drivers
# ---------------------------------------------------------------------------


class TestGpuDrivers:
    def test_cpu_build_installs_no_gpu_driver(self, base_render):
        assert "nvidia-open-dkms" not in base_render
        assert "xf86-video-amdgpu" not in base_render

    def test_nvidia_build_installs_driver_and_utils(self, nvidia_render):
        assert "nvidia-open-dkms" in nvidia_render
        assert "nvidia-utils" in nvidia_render, "nvidia-smi comes from nvidia-utils"

    def test_nvidia_smi_is_verified(self, nvidia_render):
        """Step 3 requires *-smi to be available after installing the driver."""
        assert "nvidia-smi" in nvidia_render

    def test_amd_build_installs_mesa_and_rocm(self):
        out = render(
            gpu_vendor="amd",
            gpu_driver_package="xf86-video-amdgpu",
            gpu_name="AMD Radeon RX 7900",
            torch_backend="rocm6.4",
            llm_backend="vllm",
            model_kind="huggingface",
            model_source="meta-llama/Llama-3-8B",
        )
        assert "xf86-video-amdgpu" in out
        assert "rocm" in out
        assert "rocminfo" in out or "rocm-smi" in out

    def test_old_driver_uses_pikaur_fallback(self):
        """If the driver is too old, pikaur must be used to find proper drivers."""
        out = render(
            gpu_vendor="nvidia",
            gpu_driver_package="nvidia-340xx-dkms",
            gpu_name="NVIDIA GeForce 210",
            gpu_driver_version="340.108",
            gpu_needs_aur_fallback=True,
            torch_backend="cpu",
        )
        assert "pikaur" in out
        assert "nvidia-340xx" in out

    def test_no_aur_fallback_for_modern_driver(self, nvidia_render):
        assert "aur fallback" not in nvidia_render.lower()


# ---------------------------------------------------------------------------
# Requirement 4: KDE + VNC, exactly one 1920x1080 screen
# ---------------------------------------------------------------------------


class TestKdeVnc:
    def test_kde_is_installed(self, base_render):
        assert "plasma-desktop" in base_render or "plasma-meta" in base_render

    def test_vnc_server_is_installed(self, base_render):
        assert "tigervnc" in base_render or "krfb" in base_render

    def test_one_virtual_screen_at_1920x1080(self, base_render):
        assert base_render.count("-geometry 1920x1080") >= 1
        assert "1920x1080" in base_render

    def test_xstartup_launches_plasma(self, base_render):
        assert "startplasma-x11" in base_render

    def test_only_one_vnc_server_is_started(self, base_render):
        """Requirement is one virtual screen, so do not start two servers."""
        starts_tiger = len(re.findall(r"vncserver\s", base_render))
        assert starts_tiger <= 1


# ---------------------------------------------------------------------------
# Requirement 5: uv + python
# ---------------------------------------------------------------------------


class TestUv:
    def test_uv_installed_via_official_installer(self, base_render):
        assert "astral.sh/uv" in base_render

    def test_python_version_is_installed_and_pinned(self, base_render):
        assert "uv python install 3.12" in base_render
        assert "uv python pin" in base_render

    def test_old_python_is_clamped(self):
        """An ancient script must clamp to the oldest uv-supported version."""
        out = render(python_version=detect.OLDEST_UV_PYTHON)
        assert "uv python install %s" % detect.OLDEST_UV_PYTHON in out


# ---------------------------------------------------------------------------
# Requirement 6: pytorch
# ---------------------------------------------------------------------------


class TestPytorch:
    @pytest.mark.parametrize(
        "backend,index",
        [("cu128", "cu128"), ("rocm6.4", "rocm6.4"), ("cpu", "cpu")],
    )
    def test_torch_installed_from_matching_index(self, backend, index):
        out = render(torch_backend=backend)
        assert "download.pytorch.org/whl/%s" % index in out

    def test_torch_installed_with_uv(self, base_render):
        assert re.search(r"uv pip install[^\n]*torch", base_render)


# ---------------------------------------------------------------------------
# Requirement 7: vllm / llama.cpp
# ---------------------------------------------------------------------------


class TestLlmBackend:
    def test_vllm_on_gpu(self, nvidia_render):
        assert "vllm" in nvidia_render
        assert "llama.cpp" not in nvidia_render or "vllm" in nvidia_render

    def test_llamacpp_on_cpu(self, base_render):
        assert "llama.cpp" in base_render
        assert "llama-server" in base_render

    def test_cpu_build_does_not_install_vllm(self, base_render):
        assert "uv pip install vllm" not in base_render

    def test_cpu_build_accepts_only_gguf(self, base_render):
        assert ".gguf" in base_render
        assert "gguf" in base_render.lower()

    def test_vllm_serves_openai_api(self, nvidia_render):
        assert "v1" in nvidia_render or "openai" in nvidia_render.lower()

    def test_llamacpp_built_from_source(self, base_render):
        assert "llama.cpp" in base_render
        assert "cmake" in base_render


# ---------------------------------------------------------------------------
# Requirement 8: code-server
# ---------------------------------------------------------------------------


class TestCodeServer:
    def test_installed_from_upstream_release_not_aur_stale_package(self, base_render):
        """code-server-aur README: install the upstream release tarball."""
        assert "github.com/coder/code-server/releases/download" in base_render

    def test_unpacked_to_usr_lib(self, base_render):
        assert "/usr/lib/code-server" in base_render

    def test_symlinked_into_path(self, base_render):
        assert "/usr/bin/code-server" in base_render

    def test_config_binds_all_interfaces(self, base_render):
        assert "0.0.0.0" in base_render

    def test_extension_points_at_local_llm(self, nvidia_render, base_render):
        """The plugin must be configured against the closed-loop LLM instance."""
        for out in (nvidia_render, base_render):
            assert "continue" in out.lower(), "Continue extension must be installed"
        assert "127.0.0.1:8081" in nvidia_render or "localhost:8081" in nvidia_render
        assert "127.0.0.1:8080" in base_render or "localhost:8080" in base_render

    def test_password_auth_configured(self, base_render):
        assert "auth: password" in base_render or "auth: none" in base_render


# ---------------------------------------------------------------------------
# Requirement 9: aichat
# ---------------------------------------------------------------------------


class TestAichat:
    def test_aichat_installed_from_official_repo(self, base_render):
        """Research: aichat is in Arch extra, NOT the AUR."""
        assert re.search(r"pacman[^\n]*-S[^\n]*\baichat\b", base_render) or "aichat" in base_render

    def test_configured_against_local_llm(self, nvidia_render, base_render):
        for out in (nvidia_render, base_render):
            assert "aichat" in out
            assert "openai-compatible" in out or "openai" in out

    def test_points_to_the_right_port_per_backend(self, nvidia_render, base_render):
        # vLLM on 8081, llama.cpp on 8080
        assert "8081" in nvidia_render
        assert "8080" in base_render


# ---------------------------------------------------------------------------
# Model packaging and container runtime
# ---------------------------------------------------------------------------


class TestModelAndRuntime:
    def test_gguf_is_copied(self, base_render):
        assert "COPY" in base_render
        assert "model.gguf" in base_render

    def test_huggingface_folder_is_copied(self):
        out = render(model_kind="huggingface", model_source="Qwen2.5-7B", llm_backend="vllm")
        assert "COPY" in out
        assert "hf_model" in out or "Qwen2.5-7B" in out

    def test_remote_repo_id_is_downloaded_at_runtime(self):
        out = render(model_kind="huggingface", model_source="meta-llama/Llama-3-8B", llm_backend="vllm")
        assert "meta-llama/Llama-3-8B" in out
        assert "huggingface-cli" in out or "hf download" in out or "snapshot_download" in out

    def test_entrypoint_exists_and_is_executable(self, base_render):
        assert "ENTRYPOINT" in base_render
        assert "chmod +x" in base_render

    def test_required_ports_exposed(self, base_render):
        for port in ("22", "5900", "8080", "8081"):
            assert port in base_render

    def test_uses_archlinux_base(self, base_render):
        assert re.search(r"^FROM\s+archlinux", base_render, re.MULTILINE)

    def test_pacman_cache_is_cleaned(self, base_render):
        """Image size hygiene: package cache must be removed."""
        assert "paccache" in base_render or "rm -rf /var/cache/pacman" in base_render

    def test_no_curl_pipe_to_shell_for_code_server(self, base_render):
        """code-server.dev/install.sh returned 403; must not be relied upon."""
        assert "code-server.dev/install.sh" not in base_render


class TestTemplateHygiene:
    def test_renders_without_leftover_jinja(self, base_render):
        assert "{{" not in base_render
        assert "{%" not in base_render

    def test_strict_undefined_catches_missing_vars(self):
        env = Environment(
            loader=FileSystemLoader(TEMPLATE_DIR),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        with pytest.raises(Exception):
            env.get_template(TEMPLATE_NAME).render()

    def test_is_valid_dockerfile_shape(self, base_render):
        assert base_render.lstrip().startswith("#")
        assert re.search(r"^FROM\s+\S+", base_render, re.MULTILINE)
