
# EKF-SLAM with Mahalanobis Distances

A robust SLAM (Simultaneous Localization and Mapping) implementation using an Extended Kalman Filter (EKF) and Multiple Hypothesis Tracking (MHT). This project provides both a standalone Python simulation and a Webots-based environment for testing robot navigation and mapping.

## Features

- **EKF-SLAM-MHT**: Efficiently handles data association uncertainty by maintaining multiple hypotheses.
- **Mahalanobis Distance**: Uses statistical distance for robust outlier rejection and landmark association.
- **Dual Support**: Includes a lightweight 2D Python simulation and a high-fidelity Webots simulation.
- **AprilTag Detection**: Integration with AprilTag markers for reliable visual landmarks in Webots.

## Installation

### Prerequisites

Ensure you have Python 3.x installed. For the Webots simulation, [download and install Webots](https://cyberbotics.com/doc/guide/installation-procedure).

### Dependencies

Install the required Python packages:

```bash
pip install numpy matplotlib scipy opencv-python pupil_apriltags
```

## Getting Started

### 1. Python Simulation
The standalone script simulates a robot moving in a 2D environment, visualizing its estimated trajectory and landmark positions in real-time.

```bash
python python_sim.py
```

### 2. Webots Simulation
To run the high-fidelity simulation in Webots:

1. Launch the **Webots** application.
2. Go to `File` -> `Open World...`.
3. Select `worlds/EKF_Slam.wbt` from the project directory.
4. Press the **Play** button to start the simulation.

### 3. Docker (Recommended)
You can run the entire environment using Docker. This ensures all dependencies and Webots are correctly configured.

#### Using the `run.sh` script
The project includes a convenience script `run.sh` for common tasks:

```bash
# Build the image
./run.sh build

# Run the Python simulation
./run.sh python

# Run Webots with GUI (requires X11/Wayland support)
./run.sh gui

# Run Webots headless
./run.sh headless

# Drop into a bash shell inside the container
./run.sh shell
```

#### Manual Docker Compose
Alternatively, you can use Docker Compose directly:

```bash
docker compose -f docker/docker-compose.yml up
```

## Project Structure

- `python_sim.py`: Main entry point for the 2D Python simulation.
- `source/`: Core SLAM algorithms and helper modules.
    - `EKFSlam.py`: EKF-SLAM implementation.
    - `Odometry.py`: Motion model and encoder integration.
    - `ApriltagDetector.py`: Visual landmark processing.
- `worlds/`: Webots world files.
