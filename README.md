# myarch

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/fufurobot/myarch)
[![test](https://github.com/fufurobot/myarch/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/fufurobot/myarch/actions/workflows/test.yml)

An Arch Linux AI development container that adapts itself to the machine building it.
It packs a KDE desktop over VNC, code-server for remote development, an
OpenAI-compatible local model server, and a deliberately strict privilege model —
all from one Jinja2 template rendered by host detection.

## What it gives you

| Service | Port | Notes |
| --- | --- | --- |
| SSH | 22 | password login enabled, password change forced on first login |
| VNC desktop | 5900 | KDE Plasma, exactly one 1920x1080 virtual screen |
| code-server | 8443 | Continue extension pre-pointed at the local model |
| Model server | 8080 (llama.cpp) / 8081 (vLLM) | OpenAI-compatible `/v1` |

## Requirements

- Docker, on the host you are targeting.
- For an NVIDIA GPU: a **working** driver (`nvidia-smi` must answer) and the
  NVIDIA Container Toolkit. The builder deliberately refuses to treat a card it
  can merely see in `lspci` as usable, so you never get a CUDA image that cannot
  start.
- For an AMD GPU: the in-kernel `amdgpu` module and ROCm.

## Quick start

```bash
# 1. Inspect what the builder detects. Never touches docker.
uv run python -m myarch.build --model ./examples/demo.gguf --dry-run

# 2. Build a real image (llama.cpp + cpu on a machine without a usable GPU).
uv run python -m myarch.build --model ./path/to/model.gguf --tag myarch:local

# 3. Run it.
docker run -d --name myarch \
  -p 2222:22 -p 5900:5900 -p 8443:8443 -p 8080:8080 \
  myarch:local
```

Pass a HuggingFace folder or repo id instead of a `.gguf` to get the vLLM path
(which requires a usable GPU):

```bash
uv run python -m myarch.build --model meta-llama/Llama-3-8B --tag myarch:llm
```

Point the builder at the script you actually want to run, and the image will
provide a matching Python:

```bash
uv run python -m myarch.build --model ./m.gguf --python-script ./train.py
```

Then connect:

```bash
ssh user@localhost -p 2222      # password: iamsafe (change is forced)
vncviewer localhost:5900
open http://localhost:8443      # code-server
curl localhost:8080/v1/models   # the local model server
```

## Accounts and privileges

Five accounts, each with the smallest grant that does the job. Every rule is
written to its own `/etc/sudoers.d` file and validated with `visudo -cf` during
the build, so a typo fails the build instead of silently locking or opening sudo.

| Account | Password | Allowed to sudo |
| --- | --- | --- |
| `administrator-master` | build arg | everything |
| `pikaur-manager` | `iknowtheremotecodeisdangerous` | only the exact `pikaur -Syu` invocation, with `[!-]*` args |
| `developer` | `iwillnotinstallpackagesidonotownsourcecode` | only `/usr/bin/pacman` |
| `system-manager` | `iwillnotstealotherusersdata` | only `fdisk`, `mount`, `ls` |
| `service-manager` | `iwillnotstartuntrustedservices` | only `systemctl` |
| `user` | `iamsafe` | nothing |

All five are forced to change their password on first login via `chage -d 0`.
The `[!-]*` pattern in the pikaur rule rejects any argument starting with a dash,
which blocks option injection such as `--pikaur-config=/etc/shadow`.

> These passwords are **development defaults baked into the image**. Override
> `ADMIN_MASTER_PASSWORD`, `VNC_PASSWORD` and `CODE_SERVER_PASSWORD` for anything
> that is not a throwaway container.

## Configuration

Copy `.env.example` to `.secrets.env` and fill it in. That real file is
gitignored; only the placeholders are committed.

```bash
cp .env.example .secrets.env
chmod 600 .secrets.env
set -a; . ./.secrets.env; set +a
```

## How detection works

`src/myarch/detect.py` probes the host and `src/myarch/build.py` renders
`Dockerfile.j2` from the result:

- **GPU** — `nvidia-smi` must actually answer to count as usable. A card found
  only in `lspci` is recorded as present but unusable, and the build falls back
  to CPU. Drivers older than the Arch-supported floor are resolved with `pikaur`.
- **Python** — read from the target script's `pyproject.toml`
  (`requires-python`), `.python-version`, or shebang, then clamped into the range
  `uv` can install.
- **Model** — a `.gguf` file always selects llama.cpp, because vLLM cannot read
  gguf weights; a HuggingFace folder or repo id selects vLLM when a GPU is usable.
  Staging hardlinks weights into the build context where the filesystem allows,
  so multi-gigabyte models cost no extra disk and your original file is untouched.

## Development

```bash
uv sync --group dev
uv run pytest              # 168 tests
```

The suite was written test-first: each module landed as a failing WIP commit
followed by a passing implementation commit.

Branch model: `dev` runs a fast test and is allowed to be red; `main` runs the
full suite plus a real Docker build and an in-image smoke test; pushing a `v*`
tag publishes release artifacts, and `main` deploys the rendered reference
Dockerfile to GitHub Pages.

## Layout

```
Dockerfile.j2              the Jinja2 template (the actual deliverable)
entrypoint.sh              runtime: model server, code-server, VNC, sshd
src/myarch/detect.py       host GPU / Python / model detection
src/myarch/build.py        CLI: detect, stage, render, docker build
tools/gh.py                GitHub API client used by the sprint workflow
tests/                     pytest suite
examples/demo.gguf         tiny valid GGUF fixture for CI
```
