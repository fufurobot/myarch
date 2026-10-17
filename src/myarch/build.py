#!/usr/bin/env python3
"""Host-aware image builder for the myarch Arch Linux AI container.

Detects the host GPU and Python requirements, classifies the requested model,
stages it into the build context, renders ``Dockerfile.j2`` and runs
``docker build``.

Examples
--------
    python -m myarch.build --model ./llama-3-8b.gguf
    python -m myarch.build --model meta-llama/Llama-3-8B --tag myarch:llm
    python -m myarch.build --model ./Qwen2.5-7B --python-script ./train.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import detect

#: Repository root: this file lives in ``src/myarch/``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATE_NAME = "Dockerfile.j2"
DEFAULT_CODESERVER_PORT = 8443
LLAMA_CPP_PORT = 8080
VLLM_PORT = 8081
DEFAULT_VNC_RESOLUTION = "1920x1080"


def _run(cmd: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, streaming output so long docker builds stay visible."""
    return subprocess.run(list(cmd), **kwargs)


def build_context(
    model_input: str,
    python_script: Optional[str] = None,
    *,
    vnc_resolution: str = DEFAULT_VNC_RESOLUTION,
) -> Dict[str, object]:
    """Combine host detection and model classification into a template context."""
    gpu = detect.detect_gpu()
    spec = detect.resolve_model(model_input)
    backend = detect.choose_llm_backend(gpu.vendor if gpu.usable else "none", spec)

    return {
        "gpu_vendor": gpu.vendor if gpu.usable else "none",
        "gpu_present": gpu.present,
        "gpu_usable": gpu.usable,
        "gpu_name": gpu.name,
        "gpu_driver_version": gpu.driver_version or "",
        "gpu_driver_package": gpu.driver_package if gpu.usable else "",
        "gpu_needs_aur_fallback": (
            detect.nvidia_needs_aur_fallback(gpu.driver_version) if gpu.vendor == "nvidia" else False
        ),
        "python_version": detect.python_version_for_script(python_script),
        "torch_backend": detect.torch_backend_for(gpu.vendor if gpu.usable else "none", gpu.driver_version),
        "llm_backend": backend,
        "model_kind": spec.kind,
        "model_source": spec.host_path,
        "model_repo_id": spec.repo_id or "",
        "code_server_port": DEFAULT_CODESERVER_PORT,
        "llm_port": VLLM_PORT if backend == "vllm" else LLAMA_CPP_PORT,
        "vnc_resolution": vnc_resolution,
        "enable_ssh": True,
    }


def stage_model(context: Dict[str, object], context_dir: Path) -> Dict[str, object]:
    """Copy or link the model into the docker build context.

    Weights are frequently many gigabytes, so an identical-filesystem copy is
    attempted as a hardlink first: it costs no extra disk space and the user's
    original file is never modified or removed.
    """
    staged = dict(context)
    kind = str(context["model_kind"])
    source = str(context["model_source"])
    repo_id = str(context.get("model_repo_id") or "")

    if repo_id:
        # Remote repositories are downloaded inside the image; nothing to stage.
        return staged

    models_dir = context_dir / "models"
    if kind == "gguf":
        destination = models_dir / "model.gguf"
        _place(Path(source), destination)
        staged["model_source"] = "models/model.gguf"
    else:
        destination = models_dir / "hf_model"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination, symlinks=True)
        staged["model_source"] = "models/hf_model"

    return staged


def _place(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(str(source), str(destination))
        return
    except OSError:
        pass
    shutil.copy2(str(source), str(destination))


def render_dockerfile(context: Dict[str, object], output: Path, template_dir: Optional[Path] = None) -> Path:
    """Render the Jinja2 template to ``output``."""
    directory = template_dir or PROJECT_ROOT
    env = Environment(
        loader=FileSystemLoader(str(directory)),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    rendered = env.get_template(TEMPLATE_NAME).render(**context)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return output


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="myarch",
        description="Build the Arch Linux AI development container for this host.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="path to a .gguf file, a HuggingFace model folder, or a HF repo id",
    )
    parser.add_argument(
        "--python-script",
        default=None,
        help="script whose Python version the image must provide",
    )
    parser.add_argument("--tag", default=None, help="image tag (default: myarch:<backend>-<date>)")
    parser.add_argument("--extra-tag", action="append", default=[], help="additional image tag")
    parser.add_argument("--context", default=None, help="build context directory (default: .myarch-build)")
    parser.add_argument("--output", default=None, help="where to write the rendered Dockerfile")
    parser.add_argument("--template-dir", default=None, help="directory containing Dockerfile.j2")
    parser.add_argument("--vnc-resolution", default=DEFAULT_VNC_RESOLUTION)
    parser.add_argument("--gpus", action="store_true", help="pass --gpus all to docker build")
    parser.add_argument("--no-cache", action="store_true", help="disable the docker build cache")
    parser.add_argument("--dry-run", action="store_true", help="render only; do not build")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        context = build_context(args.model, args.python_script, vnc_resolution=args.vnc_resolution)
    except FileNotFoundError as exc:
        print("error: cannot resolve model: %s" % exc, file=sys.stderr)
        return 2
    except ValueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    print("== detected build context ==")
    print(json.dumps(context, indent=2, sort_keys=True))

    if args.dry_run:
        temporary = args.output is None
        target = Path(args.output) if args.output else Path(".myarch-dry-run.Dockerfile")
        render_dockerfile(
            context,
            target,
            Path(args.template_dir) if args.template_dir else None,
        )
        sys.stdout.write("\n== rendered %s ==\n" % target)
        sys.stdout.write(target.read_text(encoding="utf-8"))
        if temporary:
            target.unlink()
        return 0

    context_dir = Path(args.context) if args.context else Path(".myarch-build")
    context_dir.mkdir(parents=True, exist_ok=True)

    try:
        staged = stage_model(context, context_dir)
    except OSError as exc:
        print("error: failed to stage model: %s" % exc, file=sys.stderr)
        return 2

    dockerfile = Path(args.output) if args.output else context_dir / "Dockerfile"
    render_dockerfile(
        staged,
        dockerfile,
        Path(args.template_dir) if args.template_dir else None,
    )
    print("wrote %s" % dockerfile)

    # The template and entrypoint must live next to the Dockerfile.
    shutil.copy2(PROJECT_ROOT / TEMPLATE_NAME, context_dir / TEMPLATE_NAME)
    shutil.copy2(PROJECT_ROOT / "entrypoint.sh", context_dir / "entrypoint.sh")

    tag = args.tag or "myarch:%s-%s" % (
        str(context["llm_backend"]).replace(".", ""),
        str(context["torch_backend"]),
    )
    cmd: List[str] = ["docker", "build", "-f", str(dockerfile), "-t", tag]
    for extra in args.extra_tag:
        cmd += ["-t", extra]
    if args.gpus or context.get("gpu_usable"):
        cmd += ["--gpus", "all"]
    if args.no_cache:
        cmd.append("--no-cache")
    cmd.append(str(context_dir))

    print("running: %s" % " ".join(cmd))
    try:
        result = _run(cmd)
    except FileNotFoundError:
        print(
            "error: docker is not installed or not on PATH; "
            "use --dry-run to inspect the rendered Dockerfile",
            file=sys.stderr,
        )
        return 2

    if result.returncode != 0:
        print("error: docker build failed with exit code %s" % result.returncode, file=sys.stderr)
        return result.returncode

    print("built image: %s" % tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
