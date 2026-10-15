#!/bin/bash
# =============================================================================
# myarch runtime entrypoint
# -----------------------------------------------------------------------------
# Starts, in order: the model server, code-server, VNC desktop and sshd, then
# hands control to the CMD (default: sleep infinity) so the container stays up.
# Every component is optional and failure-tolerant: one broken service must not
# prevent the others (and your shell) from coming up.
# =============================================================================
set -uo pipefail

log() { printf '[myarch] %s\n' "$*"; }

LLM_BACKEND="${MYARCH_LLM_BACKEND:-llama.cpp}"
MODEL_PATH="${MYARCH_MODEL_PATH:-/models/model.gguf}"
MODEL_REPO="${MYARCH_MODEL_REPO:-}"
CODE_SERVER_PORT="${CODE_SERVER_PORT:-8443}"
LLM_PORT="${LLM_PORT:-8080}"
VNC_GEOMETRY="${VNC_GEOMETRY:-1920x1080}"
VNC_DISPLAY="${VNC_DISPLAY:-:1}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"

# -----------------------------------------------------------------------------
# 1. Model server
# -----------------------------------------------------------------------------
start_llm() {
  if pgrep -f "llama-server|vllm" >/dev/null 2>&1; then
    log "model server already running"
    return 0
  fi

  if [ "${LLM_BACKEND}" = "vllm" ]; then
    log "starting vLLM on port ${LLM_PORT} (model: ${MODEL_PATH})"
    # shellcheck disable=SC1091
    /opt/venv/bin/python -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_PATH}" \
      --served-model-name local-model \
      --host 0.0.0.0 \
      --port "${LLM_PORT}" \
      >>/var/log/myarch-vllm.log 2>&1 &
  else
    if [ ! -f "${MODEL_PATH}" ]; then
      log "WARNING: no model at ${MODEL_PATH}; llama-server not started"
      return 0
    fi
    # Only gguf files are accepted on the CPU path.
    case "${MODEL_PATH}" in
      *.gguf) ;;
      *)
        log "WARNING: llama.cpp accepts only .gguf files, got ${MODEL_PATH}"
        return 0
        ;;
    esac
    log "starting llama.cpp on port ${LLM_PORT} (model: ${MODEL_PATH})"
    /usr/local/bin/llama-server \
      --model "${MODEL_PATH}" \
      --alias local-model \
      --host 0.0.0.0 \
      --port "${LLM_PORT}" \
      >>/var/log/myarch-llama.log 2>&1 &
  fi
}

# -----------------------------------------------------------------------------
# 2. Desktop + VNC (exactly one virtual screen)
# -----------------------------------------------------------------------------
start_vnc() {
  if pgrep -f "Xtigervnc|Xvnc" >/dev/null 2>&1; then
    log "vnc already running"
    return 0
  fi
  log "starting one virtual screen ${VNC_DISPLAY} at ${VNC_GEOMETRY}"
  mkdir -p /home/user/.vnc
  chown -R user:user /home/user/.vnc 2>/dev/null || true
  su - user -c "vncserver ${VNC_DISPLAY} -geometry ${VNC_GEOMETRY} -depth 24 -rfbport ${VNC_PORT}" \
    >>/var/log/myarch-vnc.log 2>&1 || log "vnc failed to start (see /var/log/myarch-vnc.log)"
}

# -----------------------------------------------------------------------------
# 2b. noVNC: browser access to the very same single virtual screen
# -----------------------------------------------------------------------------
start_novnc() {
  if pgrep -f websockify >/dev/null 2>&1; then
    log "noVNC already running"
    return 0
  fi
  if [ ! -d /usr/share/novnc ]; then
    log "noVNC not installed; skipping browser desktop"
    return 0
  fi
  log "starting noVNC on port ${NOVNC_PORT} (websockify -> ${VNC_PORT})"
  websockify --web /usr/share/novnc "${NOVNC_PORT}" "localhost:${VNC_PORT}" \
    >>/var/log/myarch-novnc.log 2>&1 &
}

# -----------------------------------------------------------------------------
# 3. code-server
# -----------------------------------------------------------------------------
start_code_server() {
  if pgrep -f "code-server" >/dev/null 2>&1; then
    log "code-server already running"
    return 0
  fi
  log "starting code-server on port ${CODE_SERVER_PORT}"
  su - user -c "code-server --bind-addr 0.0.0.0:${CODE_SERVER_PORT}" \
    >>/var/log/myarch-code-server.log 2>&1 &
}

# -----------------------------------------------------------------------------
# 4. sshd
# -----------------------------------------------------------------------------
start_sshd() {
  mkdir -p /run/sshd
  ssh-keygen -A >/dev/null 2>&1 || true
  if pgrep -x sshd >/dev/null 2>&1; then
    log "sshd already running"
    return 0
  fi
  log "starting sshd on port 22"
  /usr/sbin/sshd -D -e >>/var/log/myarch-sshd.log 2>&1 &
}

start_llm
start_vnc
start_novnc
start_code_server
start_sshd

log "ready -- ssh :22 | vnc ${VNC_DISPLAY} :${VNC_PORT} (${VNC_GEOMETRY}) | novnc :${NOVNC_PORT} | code-server :${CODE_SERVER_PORT} | ${LLM_BACKEND} :${LLM_PORT}"

# Hand off to the container command (default: keep running).
if [ "$#" -eq 0 ]; then
  set -- sleep infinity
fi
exec "$@"
