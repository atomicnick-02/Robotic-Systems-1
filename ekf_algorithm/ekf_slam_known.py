#!/usr/bin/env python3

"""
EKF SLAM Implementation with Known Correspondences
This module implements Extended Kalman Filter (EKF) SLAM algorithm 
with known data association.
"""

import numpy as np
import matplotlib.pyplot as plt


def wrap_angle(angle: float) -> float:
    """
    Normalize angle to be between -π and π.
    
    Args:
        angle: Input angle in radians
    Returns:
        Normalized angle in radians
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def initialize_ekf() -> tuple:
    """
    Initialize EKF state and covariance matrices.
    
    Returns:
        tuple: (mu, Sigma) Initial state vector and covariance matrix
    """
    mu = np.zeros(3)  # Robot state [x, y, theta]
    Sigma = np.zeros((3, 3))  # Initial covariance matrix
    return mu, Sigma


def ekf_slam_known_correspondences(mu: np.ndarray, 
                                 Sigma: np.ndarray,
                                 u: list,
                                 z: list,
                                 c: list,
                                 R: np.ndarray,
                                 Q: np.ndarray,
                                 dt: float) -> tuple:
    """
    Perform one step of EKF SLAM algorithm with known correspondences.
    
    Args:
        mu: Current state mean
        Sigma: Current state covariance
        u: Control input [v, w] (linear and angular velocity)
        z: List of measurements [range, bearing]
        c: List of landmark correspondences
        R: Motion noise covariance
        Q: Measurement noise covariance
        dt: Time step
        
    Returns:
        tuple: (mu_bar, Sigma_bar) Updated state and covariance
    """
    # Get number of landmarks and create augmented state matrix
    n_landmarks = (len(mu) - 3) // 2
    Fx = np.hstack((np.eye(3), np.zeros((3, 2 * n_landmarks))))

    # Extract control inputs
    v, w = u
    theta = mu[2]
    dtheta = w * dt

    # Prediction step - update robot pose
    if abs(w) > 1e-6:  # Handle non-zero angular velocity
        dx = -v / w * np.sin(theta) + v / w * np.sin(theta + dtheta)
        dy = v / w * np.cos(theta) - v / w * np.cos(theta + dtheta)
    else:  # Handle straight line motion
        dx = v * dt * np.cos(theta)
        dy = v * dt * np.sin(theta)

    # Update state prediction
    mu_bar = mu.copy()
    mu_bar[0] += dx
    mu_bar[1] += dy
    mu_bar[2] = wrap_angle(mu_bar[2] + dtheta)

    # Calculate Jacobian of motion model
    Gt = np.zeros((3, 3))
    if abs(w) > 1e-6:
        Gt[0, 2] = v / w * (np.cos(theta) - np.cos(theta + w * dt))
        Gt[1, 2] = v / w * (np.sin(theta) - np.sin(theta + w * dt))
    else:
        Gt[0, 2] = -v * dt * np.sin(theta)
        Gt[1, 2] = v * dt * np.cos(theta)

    # Update state covariance prediction
    Gt = np.eye(len(mu)) + Fx.T @ Gt @ Fx
    Sigma_bar = Gt @ Sigma @ Gt.T + Fx.T @ R @ Fx

    # Process each measurement
    for i in range(len(z)):
        r, phi = z[i]
        j = c[i]
        lm_index = 3 + 2 * j

        # Initialize new landmark if not seen before
        if len(mu_bar) <= lm_index + 1:
            mu_bar = np.append(mu_bar, [
                mu_bar[0] + r * np.cos(phi + mu_bar[2]),
                mu_bar[1] + r * np.sin(phi + mu_bar[2])
            ])
            new_Sigma = np.zeros((len(mu_bar), len(mu_bar)))
            new_Sigma[:Sigma_bar.shape[0], :Sigma_bar.shape[1]] = Sigma_bar
            new_Sigma[lm_index:, lm_index:] = np.eye(2) * 1e4
            Sigma_bar = new_Sigma

    # Update step
    Stemp = np.zeros((len(mu_bar), len(mu_bar)))

    for i in range(len(z)):
        r, phi = z[i]
        j = c[i]
        lm_index = 3 + 2 * j

        # Calculate innovation
        delta = mu_bar[lm_index:lm_index+2] - mu_bar[0:2]
        q = delta.T @ delta
        z_hat = np.array([
            np.sqrt(q),
            wrap_angle(np.arctan2(delta[1], delta[0]) - mu_bar[2])
        ])

        # Calculate measurement Jacobian
        delta_x, delta_y = delta
        Fxj = np.zeros((5, len(mu_bar)))
        Fxj[0:3, 0:3] = np.eye(3)
        Fxj[3, lm_index] = 1
        Fxj[4, lm_index+1] = 1

        H = (1 / q) * np.array([
            [-np.sqrt(q) * delta_x, -np.sqrt(q) * delta_y, 0,
             np.sqrt(q) * delta_x, np.sqrt(q) * delta_y],
            [delta_y, -delta_x, -q, -delta_y, delta_x]
        ]) @ Fxj

        # Kalman update
        S = H @ Sigma_bar @ H.T + Q
        K = Sigma_bar @ H.T @ np.linalg.inv(S)

        z_actual = np.array([r, phi])
        dz = z_actual - z_hat
        dz[1] = wrap_angle(dz[1])
        
        mu_bar = mu_bar + K @ dz
        mu_bar[2] = wrap_angle(mu_bar[2])
        Stemp += K @ H

    # Update covariance
    Sigma_bar = (np.eye(len(mu_bar)) - Stemp) @ Sigma_bar

    return mu_bar, Sigma_bar


def run_ekf_slam_test():
    """
    Run a test simulation of the EKF SLAM algorithm.
    
    Returns:
        tuple: (path, true_landmarks, estimated_landmarks)
    """
    # Initialize system
    mu, Sigma = initialize_ekf()
    dt = 1.0
    R = np.diag([0.1, 0.1, np.deg2rad(5)]) ** 2  # Motion noise
    Q = np.diag([1.0, np.deg2rad(20)]) ** 2      # Measurement noise

    # Define true landmark positions
    landmarks = np.array([[5, 10], [10, 0], [15, 15]])

    # Storage for results
    path = [mu[:2].copy()]
    est_landmarks_over_time = []

    # Main simulation loop
    for t in range(20):
        # Control input (constant velocity and turn rate)
        u = [1.0, np.deg2rad(10)]

        # Generate measurements
        z = []
        c = []
        for j, (lx, ly) in enumerate(landmarks):
            dx = lx - mu[0]
            dy = ly - mu[1]
            r = np.sqrt(dx**2 + dy**2) + np.random.normal(0, np.sqrt(Q[0, 0]))
            phi = wrap_angle(np.arctan2(dy, dx) - mu[2] + 
                           np.random.normal(0, np.sqrt(Q[1, 1])))
            z.append([r, phi])
            c.append(j)

        # Update filter
        mu, Sigma = ekf_slam_known_correspondences(mu, Sigma, u, z, c, R, Q, dt)
        path.append(mu[:2].copy())
        est_landmarks = np.array(mu[3:]).reshape(-1, 2)
        est_landmarks_over_time.append(est_landmarks)

    return np.array(path), landmarks, est_landmarks_over_time[-1]


def plot_results(path, true_landmarks, estimated_landmarks):
    """
    Plot the results of the SLAM algorithm.
    
    Args:
        path: Robot trajectory
        true_landmarks: True landmark positions
        estimated_landmarks: Estimated landmark positions
    """
    plt.figure(figsize=(8, 6))
    plt.plot(path[:, 0], path[:, 1], 'b-o', label='Estimated Path')
    plt.scatter(true_landmarks[:, 0], true_landmarks[:, 1], 
                c='g', marker='^', label='True Landmarks')
    plt.scatter(estimated_landmarks[:, 0], estimated_landmarks[:, 1], 
                c='r', marker='x', label='Estimated Landmarks')
    plt.legend()
    plt.title('EKF-SLAM with Known Correspondences')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    path, true_landmarks, estimated_landmarks = run_ekf_slam_test()
    plot_results(path, true_landmarks, estimated_landmarks)
