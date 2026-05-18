#!/usr/bin/env bash
# ============================================================
# run.sh — convenience wrapper for Arch Linux
#
# Usage:
#   ./run.sh build           # build the image
#   ./run.sh gui             # launch Webots with GUI
#   ./run.sh headless        # launch Webots headless
#   ./run.sh python          # run python_sim.py
#   ./run.sh shell           # drop into a bash shell
#   ./run.sh clean           # remove containers + volumes
# ============================================================
set -euo pipefail

# Support both "docker compose" (plugin) and "docker-compose" (standalone)
if docker compose version &>/dev/null; then
    COMPOSE="docker compose -f docker/docker-compose.yml"
elif command -v docker-compose &>/dev/null; then
    COMPOSE="docker-compose -f docker/docker-compose.yml"
else
    echo "ERROR: neither 'docker compose' nor 'docker-compose' found. Install one and retry."
    exit 1
fi
IMAGE="ekf-slam-webots:latest"

case "${1:-help}" in

  # ---------- Build ----------
  build)
    echo ">>> Building ${IMAGE}..."
    docker build -t "${IMAGE}" .
    ;;

  # ---------- GUI mode ----------
  gui)
    echo ">>> Granting X11 access to local Docker connections..."
    xhost +local:docker 2>/dev/null || true

    echo ">>> Starting Webots (GUI)..."
    ${COMPOSE} --profile gui up --build

    echo ">>> Revoking X11 access..."
    xhost -local:docker 2>/dev/null || true
    ;;

  # ---------- Headless mode ----------
  headless)
    echo ">>> Starting Webots (headless / Xvfb)..."
    ${COMPOSE} --profile headless up --build
    ;;

  # ---------- Python sim ----------
  python)
    echo ">>> Granting X11 access..."
    xhost +local:docker 2>/dev/null || true

    echo ">>> Running controller standalone..."
    ${COMPOSE} --profile python up --build

    xhost -local:docker 2>/dev/null || true
    ;;

  # ---------- Interactive shell ----------
  shell)
    echo ">>> Opening bash shell in container..."
    xhost +local:docker 2>/dev/null || true

    docker run --rm -it \
      --name webots_shell \
      -e DISPLAY="${DISPLAY:-:0}" \
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
      -v "$(pwd)":/home/simuser/ekf_slam:rw \
      --device /dev/dri:/dev/dri 2>/dev/null || true \
      "${IMAGE}" bash

    xhost -local:docker 2>/dev/null || true
    ;;

  # ---------- Clean up ----------
  clean)
    echo ">>> Stopping and removing containers..."
    ${COMPOSE} --profile gui --profile headless --profile python down -v
    echo ">>> Done."
    ;;

  # ---------- Help ----------
  *)
    echo ""
    echo "  Webots EKF-SLAM — Docker runner for Arch Linux"
    echo ""
    echo "  Usage: ./run.sh <command>"
    echo ""
    echo "  Commands:"
    echo "    build      Build the Docker image"
    echo "    gui        Launch Webots with GUI (X11 forwarding)"
    echo "    headless   Launch Webots headless (Xvfb)"
    echo "    python     Run the 2D Python simulation"
    echo "    shell      Open an interactive bash shell"
    echo "    clean      Remove all containers and volumes"
    echo ""
    ;;
esac