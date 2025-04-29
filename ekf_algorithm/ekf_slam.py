import numpy as np
import matplotlib.pyplot as plt

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def motion_model(mu, u, dt):
    x, y, theta = mu[0:3]
    v, w = u
    theta_new = wrap_angle(theta + w * dt)
    x_new = x + v * dt * np.cos(theta)
    y_new = y + v * dt * np.sin(theta)
    mu[0:3] = [x_new, y_new, theta_new]
    return mu

def observation_model(mu, landmark_idx):
    x, y, theta = mu[0:3]
    lx, ly = mu[landmark_idx:landmark_idx+2]
    dx = lx - x
    dy = ly - y
    r = np.sqrt(dx**2 + dy**2)
    phi = wrap_angle(np.arctan2(dy, dx) - theta)
    return np.array([r, phi])

def compute_jacobians(mu, landmark_idx):
    x, y, theta = mu[0:3]
    lx, ly = mu[landmark_idx:landmark_idx+2]
    dx = lx - x
    dy = ly - y
    q = dx**2 + dy**2
    sqrt_q = np.sqrt(q)

    H = np.zeros((2, len(mu)))
    H[0, 0] = -dx / sqrt_q
    H[0, 1] = -dy / sqrt_q
    H[0, landmark_idx] = dx / sqrt_q
    H[0, landmark_idx + 1] = dy / sqrt_q

    H[1, 0] = dy / q
    H[1, 1] = -dx / q
    H[1, 2] = -1
    H[1, landmark_idx] = -dy / q
    H[1, landmark_idx + 1] = dx / q

    return H

def mahalanobis(z, z_pred, S):
    dz = z - z_pred
    dz[1] = wrap_angle(dz[1])
    return dz.T @ np.linalg.inv(S) @ dz

def ekf_slam(mu, Sigma, u, z_list, Q, R, dt, mahalanobis_thresh=5.99):
    mu = motion_model(mu, u, dt)
    Sigma[:3, :3] += R  # Add motion noise

    for z in z_list:
        r, phi = z
        x, y, theta = mu[0:3]
        lx = x + r * np.cos(theta + phi)
        ly = y + r * np.sin(theta + phi)
        z_new = np.array([r, phi])

        matched = False
        for i in range(3, len(mu), 2):
            z_pred = observation_model(mu, i)
            H = compute_jacobians(mu, i)
            S = H @ Sigma @ H.T + Q
            d = mahalanobis(z_new, z_pred, S)

            if d < mahalanobis_thresh:
                dz = z_new - z_pred
                dz[1] = wrap_angle(dz[1])
                K = Sigma @ H.T @ np.linalg.inv(S)
                mu = mu + K @ dz
                mu[2] = wrap_angle(mu[2])
                Sigma = (np.eye(len(mu)) - K @ H) @ Sigma
                matched = True
                break

        if not matched:
            # Add new landmark
            mu = np.append(mu, [lx, ly])
            n = len(mu)
            Sigma_new = np.zeros((n, n))
            Sigma_new[:n-2, :n-2] = Sigma
            Sigma_new[n-2:, n-2:] = np.eye(2) * 1000  # High initial uncertainty
            Sigma = Sigma_new

    return mu, Sigma

# Simulate EKF-SLAM
np.random.seed(42)

# Initial state
mu = np.array([0, 0, 0])  # x, y, theta
Sigma = np.eye(3) * 0.01

# Landmarks in the world
landmarks = np.array([[5, 10], [10, 0], [15, 15]])

# Noise parameters
Q = np.diag([0.5, np.deg2rad(10)]) ** 2  # Measurement noise
R = np.diag([0.1, 0.1, np.deg2rad(5)]) ** 2  # Motion noise

# Store robot path and estimated landmarks
path = [mu[:2].copy()]
estimated_landmarks = []

# Run for a few steps
for t in range(10):
    # Simulate control input
    u = [1.0, np.deg2rad(10)]  # v, w
    dt = 1.0

    # Simulate observations (noisy)
    x, y, theta = mu[0:3]
    z_list = []
    for lx, ly in landmarks:
        dx = lx - x
        dy = ly - y
        r = np.sqrt(dx**2 + dy**2) + np.random.normal(0, np.sqrt(Q[0, 0]))
        phi = wrap_angle(np.arctan2(dy, dx) - theta + np.random.normal(0, np.sqrt(Q[1, 1])))
        z_list.append([r, phi])

    mu, Sigma = ekf_slam(mu, Sigma, u, z_list, Q, R, dt)
    path.append(mu[:2].copy())

# Plot results
path = np.array(path)
est_landmarks = np.array(mu[3:]).reshape(-1, 2)

plt.figure(figsize=(8, 6))
plt.plot(path[:, 0], path[:, 1], 'b-o', label='Estimated Path')
plt.scatter(landmarks[:, 0], landmarks[:, 1], c='g', marker='^', label='True Landmarks')
plt.scatter(est_landmarks[:, 0], est_landmarks[:, 1], c='r', marker='x', label='Estimated Landmarks')
plt.legend()
plt.title('EKF-SLAM without Correspondence (Mahalanobis Thresholding)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.show()
