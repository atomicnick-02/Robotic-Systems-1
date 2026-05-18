# ============================================================
# Webots EKF-SLAM Docker Image
# Base: Ubuntu 22.04 (jammy) — best Webots compatibility
# Supports: X11 forwarding (GUI) + headless (Xvfb) modes
# ============================================================
FROM ubuntu:22.04

# ---------- Build args ----------
ARG WEBOTS_VERSION=R2025a
ARG WEBOTS_DEB_VERSION=2025a
ARG DEBIAN_FRONTEND=noninteractive

# ---------- Labels ----------
LABEL maintainer="ekf-slam"
LABEL description="Webots EKF-SLAM simulation environment"
LABEL webots.version="${WEBOTS_VERSION}"

# ---------- Base system packages ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core utils
    ca-certificates \
    curl \
    wget \
    git \
    unzip \
    sudo \
    # Python
    python3 \
    python3-pip \
    python3-dev \
    # X11 / display (needed for Webots GUI + Xvfb headless)
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxfixes3 \
    libxcursor1 \
    libxi6 \
    libxinerama1 \
    libxss1 \
    libxtst6 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libegl1-mesa \
    # Virtual framebuffer for headless mode
    xvfb \
    x11-utils \
    # OpenCV system deps
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    # Misc Webots runtime deps
    libdbus-1-3 \
    libpulse0 \
    libasound2 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxshmfence1 \
    libgbm1 \
    # Qt for Matplotlib interactive windows
    python3-pyqt5 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Install Webots ----------
# Downloads the official .deb from GitHub releases
RUN wget -q "https://github.com/cyberbotics/webots/releases/download/${WEBOTS_VERSION}/webots_${WEBOTS_DEB_VERSION}_amd64.deb" \
        -O /tmp/webots.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends /tmp/webots.deb \
    && rm /tmp/webots.deb \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python dependencies ----------
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    scipy \
    opencv-python-headless \
    pupil_apriltags

# ---------- Create non-root user (avoids Webots root warnings) ----------
RUN useradd -ms /bin/bash simuser \
    && echo "simuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER simuser
WORKDIR /home/simuser

# ---------- Copy project into image ----------
# If you prefer bind-mounting at runtime, comment this COPY out
# and use the volume mount in docker-compose instead.
COPY --chown=simuser:simuser . /home/simuser/ekf_slam/

WORKDIR /home/simuser/ekf_slam

# ---------- Environment ----------
ENV WEBOTS_HOME=/usr/local/webots
ENV PATH="${WEBOTS_HOME}/bin:${PATH}"

# Matplotlib config directory to avoid permission issues
ENV MPLCONFIGDIR=/tmp/matplotlib-cache

# Webots built-in Python controller bindings
ENV PYTHONPATH="${WEBOTS_HOME}/lib/controller/python"

# ── Repo-specific paths ──────────────────────────────────────
# controller root → so `from source.EKFSlam import ...` resolves correctly
# (source/ lives at controllers/my_first_controller0/source/)
ENV PYTHONPATH="${PYTHONPATH}:/home/simuser/ekf_slam/Robotic_Systems/controllers/my_first_controller0"

# Webots resolves PROTO files relative to the project directory
ENV WEBOTS_PROJECT_DIR=/home/simuser/ekf_slam/Robotic_Systems

# Suppress Qt platform warnings in X11 mode
ENV QT_X11_NO_MITSHM=1
# Default display (overridable at runtime)
ENV DISPLAY=:0

# ---------- Entrypoint (inlined — avoids .dockerignore issues) ----------
RUN cat > /home/simuser/entrypoint.sh << 'EOF'
#!/usr/bin/env bash
set -e

DISPLAY_MODE="${DISPLAY_MODE:-GUI}"

if [[ "${DISPLAY_MODE}" == "HEADLESS" ]]; then
    echo "[entrypoint] Starting Xvfb virtual display on :99"
    Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset &
    XVFB_PID=$!
    export DISPLAY=:99
    sleep 1
    trap "kill ${XVFB_PID} 2>/dev/null || true" EXIT
else
    echo "[entrypoint] Using host X display: ${DISPLAY}"
    if [[ -z "${DISPLAY}" ]]; then
        echo "[entrypoint] WARNING: DISPLAY not set. Run: xhost +local:docker"
    fi
fi

echo "[entrypoint] Webots: $(webots --version 2>/dev/null || echo 'unknown')"
echo "[entrypoint] Python: $(python3 --version)"
echo "[entrypoint] Executing: $*"
exec "$@"
EOF
RUN chmod +x /home/simuser/entrypoint.sh

ENTRYPOINT ["/home/simuser/entrypoint.sh"]
# Default: launch Webots with the EKF world
CMD ["webots", "--mode=realtime", "/home/simuser/ekf_slam/Robotic_Systems/worlds/EKF_Slam.wbt"]