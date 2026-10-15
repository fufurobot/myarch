#!/usr/bin/env bash
# =============================================================================
# GitHub Codespaces / devcontainer setup for myarch
# -----------------------------------------------------------------------------
# This script must not assume a particular Linux distribution: Codespaces
# images vary. It detects the package manager and always prefers the official
# upstream installer for each tool.
#
# It is idempotent: running it twice is safe and fast.
# =============================================================================
set -euo pipefail

log() { printf '\n[post-create] %s\n' "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

# -----------------------------------------------------------------------------
# uv (official installer; does not assume a distro)
# -----------------------------------------------------------------------------
if have uv; then
  log "uv already installed: $(uv --version)"
else
  log "installing uv via the official installer"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer writes to ~/.local/bin, which may not be on PATH yet.
  export PATH="${HOME}/.local/bin:${PATH}"
  if ! have uv; then
    log "ERROR: uv installation failed"
    exit 1
  fi
  log "installed $(uv --version)"
fi
export PATH="${HOME}/.local/bin:${PATH}"

# Persist PATH for later shells (Codespaces starts a fresh shell per terminal).
if ! grep -q 'myarch: uv path' "${HOME}/.bashrc" 2>/dev/null; then
  printf '\n# myarch: uv path\nexport PATH="${HOME}/.local/bin:${PATH}"\n' >> "${HOME}/.bashrc"
fi

# -----------------------------------------------------------------------------
# Python toolchain + dependencies
# -----------------------------------------------------------------------------
log "installing the pinned Python and project dependencies"
# --group dev pulls pytest/pytest-cov; --frozen respects the committed lockfile.
if ! uv sync --group dev --frozen; then
  log "frozen sync failed (lockfile drift?); retrying without --frozen"
  uv sync --group dev
fi

# -----------------------------------------------------------------------------
# Optional system tools, only when a known package manager exists
# -----------------------------------------------------------------------------
if have apt-get && ! have shellcheck; then
  log "installing shellcheck via apt-get"
  sudo apt-get update -qq && sudo apt-get install -y -qq shellcheck || log "shellcheck install skipped"
elif have dnf && ! have shellcheck; then
  log "installing shellcheck via dnf"
  sudo dnf install -y -q shellcheck || log "shellcheck install skipped"
elif have pacman && ! have shellcheck; then
  log "installing shellcheck via pacman"
  sudo pacman -S --noconfirm --needed shellcheck || log "shellcheck install skipped"
fi

# -----------------------------------------------------------------------------
# Sanity checks so a broken codespace fails loudly at create time
# -----------------------------------------------------------------------------
log "running a quick self-check"
uv run python -c "import myarch, jinja2; print('myarch package importable')"
uv run python -c "
from myarch import detect
info = detect.detect_host()
print('detected host:', {k: info[k] for k in ('gpu_vendor', 'gpu_present', 'gpu_usable', 'torch_backend')})
"

log "running the test suite"
uv run pytest -q

log "done. Try: uv run python -m myarch.build --model <model> --dry-run"
