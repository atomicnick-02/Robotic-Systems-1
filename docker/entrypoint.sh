#!/usr/bin/env bash
# ============================================================
# entrypoint.sh — handles GUI (X11) and headless (Xvfb) modes
# Usage controlled by DISPLAY_MODE env var:
#   GUI      → forward to host X server (default)
#   HEADLESS → spin up Xvfb virtual display
# ============================================================
set -e

DISPLAY_MODE="${DISPLAY_MODE:-GUI}"

# ---------- Headless mode: start virtual framebuffer ----------
if [[ "${DISPLAY_MODE}" == "HEADLESS" ]]; then
    echo "[entrypoint] Starting Xvfb virtual display on :99"
    Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset &
    XVFB_PID=$!
    export DISPLAY=:99

    # Give Xvfb a moment to initialize
    sleep 1

    # Cleanup Xvfb on exit
    trap "kill ${XVFB_PID} 2>/dev/null || true" EXIT

# ---------- GUI mode: use host X server ----------
else
    echo "[entrypoint] Using host X display: ${DISPLAY}"

    # Warn if DISPLAY not set
    if [[ -z "${DISPLAY}" ]]; then
        echo "[entrypoint] WARNING: DISPLAY is not set. X11 forwarding may not work."
        echo "[entrypoint] On Arch Linux, run: xhost +local:docker before starting."
    fi
fi

echo "[entrypoint] Webots version: $(webots --version 2>/dev/null || echo 'unknown')"
echo "[entrypoint] Python: $(python3 --version)"
echo "[entrypoint] Executing: $*"
echo "---"

exec "$@"