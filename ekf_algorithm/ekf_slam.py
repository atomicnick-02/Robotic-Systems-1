"""
EKF SLAM Implementation with Unknown Correspondences
This module implements Extended Kalman Filter (EKF) SLAM algorithm for robot localization
and landmark mapping with unknown data association.
"""

import numpy as np
import matplotlib.pyplot as plt


def wrap_angle(angle: float) -> float:
    """
    Normalize angle to [-π, π] range.
    
    Args:
        angle: Input angle in radians
    
    Returns:
        Normalized angle in radians
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def initialize_ekf() -> tuple:
    """
    Initialize EKF state vector and covariance matrix.
    
    Returns:
        tuple: (mu, Sigma) Initial state vector and covariance matrix
    """
    mu = np.zeros(3)  # Initial state [x, y, theta]
    sigma = np.zeros((3, 3))  # Initial covariance matrix
    return mu, sigma


def ekf_slam_unknown_correspondences(
    mu: np.ndarray,
    sigma: np.ndarray,
    u: list,
    z: list,
    R: np.ndarray,
    Q: np.ndarray,
    dt: float,
    alpha_threshold: float = 5.99
) -> tuple:
    """
    Implement EKF SLAM with unknown correspondences.
    
    Args:
        mu: State vector [robot_pose, landmarks_positions]
        sigma: Covariance matrix
        u: Control input [v, w] (linear and angular velocity)
        z: List of measurements [range, bearing]
        R: Motion noise covariance
        Q: Measurement noise covariance
        dt: Time step
        alpha_threshold: Mahalanobis distance threshold for new landmarks
    
    Returns:
        tuple: (mu_bar, sigma_bar) Updated state and covariance
    """
    n_landmarks = (len(mu) - 3) // 2
    Fx = np.hstack([np.eye(3), np.zeros((3, 2 * n_landmarks))])

    # Extract control inputs
    v, w = u
    theta = mu[2]
    dtheta = w * dt

    # Compute motion update
    if abs(w) > 1e-6:
        # Circular motion
        dx = -v / w * np.sin(theta) + v / w * np.sin(theta + dtheta)
        dy = v / w * np.cos(theta) - v / w * np.cos(theta + dtheta)
    else:
        # Straight line motion
        dx = v * dt * np.cos(theta)
        dy = v * dt * np.sin(theta)

    # Prediction step
    mu_bar = mu.copy()
    mu_bar[0] += dx
    mu_bar[1] += dy
    mu_bar[2] = wrap_angle(mu_bar[2] + dtheta)

    # Compute Jacobian of motion model
    Gx = (np.array([
        [0, 0, v/w * np.cos(theta) - v/w * np.cos(theta + dtheta)],
        [0, 0, v/w * np.sin(theta) - v/w * np.sin(theta + dtheta)],
        [0, 0, 0]
    ]) if abs(w) > 1e-6 else np.array([
        [0, 0, -v * dt * np.sin(theta)],
        [0, 0, v * dt * np.cos(theta)],
        [0, 0, 0]
    ]))

    # Update state covariance
    G = np.eye(len(mu)) + Fx.T @ Gx @ Fx
    sigma_bar = G @ sigma @ G.T + Fx.T @ R @ Fx

    # Process each measurement
    for z_i in z:
        r_i, phi_i = z_i
        z_measurement = np.array([r_i, phi_i])

        # Data association
        min_mahalanobis = float('inf')
        best_dz = None
        best_H = None
        best_K = None

        # Check all existing landmarks
        n_total = (len(mu_bar) - 3) // 2
        for j in range(n_total):
            lm_index = 3 + 2 * j
            delta = mu_bar[lm_index:lm_index+2] - mu_bar[0:2]
            q = delta.T @ delta
            sqrt_q = np.sqrt(q)
            
            # Predicted measurement
            z_hat = np.array([
                sqrt_q,
                wrap_angle(np.arctan2(delta[1], delta[0]) - mu_bar[2])
            ])
            dz = z_measurement - z_hat
            dz[1] = wrap_angle(dz[1])

            # Compute measurement Jacobian
            Fxj = np.zeros((5, len(mu_bar)))
            Fxj[0:3, 0:3] = np.eye(3)
            Fxj[3, lm_index] = 1
            Fxj[4, lm_index+1] = 1

            H = (1 / q) * np.array([
                [-sqrt_q * delta[0], -sqrt_q * delta[1], 0, sqrt_q * delta[0], sqrt_q * delta[1]],
                [delta[1], -delta[0], -q, -delta[1], delta[0]]
            ]) @ Fxj

            # Compute Mahalanobis distance
            S = H @ sigma_bar @ H.T + Q
            mahalanobis = dz.T @ np.linalg.inv(S) @ dz

            if mahalanobis < min_mahalanobis:
                min_mahalanobis = mahalanobis
                best_dz = dz
                best_H = H
                best_K = sigma_bar @ H.T @ np.linalg.inv(S)

        if min_mahalanobis > alpha_threshold:
            # Add new landmark
            lx = mu_bar[0] + r_i * np.cos(phi_i + mu_bar[2])
            ly = mu_bar[1] + r_i * np.sin(phi_i + mu_bar[2])
            mu_bar = np.append(mu_bar, [lx, ly])

            # Expand covariance matrix
            new_sigma = np.zeros((len(mu_bar), len(mu_bar)))
            new_sigma[:sigma_bar.shape[0], :sigma_bar.shape[1]] = sigma_bar
            new_sigma[-2:, -2:] = np.eye(2) * 1e2
            sigma_bar = new_sigma

            print(f"✅ New landmark added at ({lx:.2f}, {ly:.2f})")
        else:
            # Update existing landmark
            mu_bar = mu_bar + best_K @ best_dz
            mu_bar[2] = wrap_angle(mu_bar[2])
            sigma_bar = (np.eye(len(mu_bar)) - best_K @ best_H) @ sigma_bar

    return mu_bar, sigma_bar


def run_ekf_slam_test_unknown_correspondences():
    """Run a test simulation of the EKF SLAM algorithm."""
    # Initialize parameters
    mu, sigma = initialize_ekf()
    dt = 1.0
    R = np.diag([0.1, 0.1, np.deg2rad(5)]) ** 2  # Motion noise
    Q = np.diag([1.0, np.deg2rad(20)]) ** 2  # Measurement noise

    # Set up simulation environment
    landmarks = np.array([[5, 10], [10, 0], [15, 15]])
    path = [mu[:2].copy()]
    est_landmarks_over_time = []

    # Run simulation
    for _ in range(20):
        u = [1.0, np.deg2rad(10)]  # Control input
        z = []
        
        # Generate measurements
        for lx, ly in landmarks:
            dx = lx - mu[0]
            dy = ly - mu[1]
            r = np.sqrt(dx**2 + dy**2) + np.random.normal(0, np.sqrt(Q[0, 0]))
            phi = wrap_angle(np.arctan2(dy, dx) - mu[2] + 
                           np.random.normal(0, np.sqrt(Q[1, 1])))
            z.append([r, phi])

        # Update SLAM
        mu, sigma = ekf_slam_unknown_correspondences(mu, sigma, u, z, R, Q, dt)
        path.append(mu[:2].copy())
        est_landmarks = np.array(mu[3:]).reshape(-1, 2)
        est_landmarks_over_time.append(est_landmarks)

    # Visualize results
    _plot_results(np.array(path), landmarks, est_landmarks_over_time[-1])


def _plot_results(path: np.ndarray, true_landmarks: np.ndarray, 
                 estimated_landmarks: np.ndarray):
    """
    Plot the results of the SLAM algorithm.
    
    Args:
        path: Robot trajectory
        true_landmarks: Ground truth landmark positions
        estimated_landmarks: Estimated landmark positions
    """
    plt.figure(figsize=(8, 6))
    plt.plot(path[:, 0], path[:, 1], 'b-o', label='Estimated Path')
    plt.scatter(true_landmarks[:, 0], true_landmarks[:, 1], 
               c='g', marker='^', label='True Landmarks')
    
    if len(estimated_landmarks) > 0:
        plt.scatter(estimated_landmarks[:, 0], estimated_landmarks[:, 1], 
                   c='r', marker='x', label='Estimated Landmarks')
    
    plt.legend()
    plt.title('EKF-SLAM with Unknown Correspondences')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    run_ekf_slam_test_unknown_correspondences()
