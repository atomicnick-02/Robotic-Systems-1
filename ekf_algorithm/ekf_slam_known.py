import numpy as np
import matplotlib.pyplot as plt

# Helper functions
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Initialize variables
def initialize_ekf():
    mu = np.zeros(3)  # Initial state [x, y, theta]
    Sigma = np.eye(3) * 1e-3
    return mu, Sigma

# Main EKF SLAM with known correspondences
def ekf_slam_known_correspondences(mu, Sigma, u, z, c, R, Q, dt):
    n_landmarks = (len(mu) - 3) // 2
    Fx = np.hstack((np.eye(3), np.zeros((3, 2 * n_landmarks))))

    v, w = u
    theta = mu[2]
    if abs(w) > 1e-6:
        dx = -v/w * np.sin(theta) + v/w * np.sin(theta + w*dt)
        dy = v/w * np.cos(theta) - v/w * np.cos(theta + w*dt)
    else: 
        dx = v * dt * np.cos(theta)
        dy = v * dt * np.sin(theta)
    dtheta = w * dt
    mu_bar = mu.copy()
    mu_bar[0] += dx
    mu_bar[1] += dy
    mu_bar[2] = wrap_angle(mu_bar[2] + dtheta)

    Gt = np.eye(len(mu))
    if abs(w) > 1e-6:
        Gt[0,2] = -v/w * np.cos(theta) + v/w * np.cos(theta + w*dt)
        Gt[1,2] = -v/w * np.sin(theta) + v/w * np.sin(theta + w*dt)
    else:
        Gt[0,2] = -v * dt * np.sin(theta)
        Gt[1,2] = v * dt * np.cos(theta)

    Sigma_bar = Gt @ Sigma @ Gt.T + Fx.T @ R @ Fx

    for i in range(len(z)):
        r, phi = z[i]
        j = c[i]

        lm_index = 3 + 2*j
        if len(mu_bar) <= lm_index + 1:
            # Landmark never seen before
            mu_bar = np.append(mu_bar, [mu_bar[0] + r * np.cos(phi + mu_bar[2]),
                                        mu_bar[1] + r * np.sin(phi + mu_bar[2])])
            new_Sigma = np.zeros((len(mu_bar), len(mu_bar)))
            new_Sigma[:Sigma_bar.shape[0], :Sigma_bar.shape[1]] = Sigma_bar
            new_Sigma[lm_index:, lm_index:] = np.eye(2) * 1e3
            Sigma_bar = new_Sigma

        delta = mu_bar[lm_index:lm_index+2] - mu_bar[0:2]
        q = delta.T @ delta
        z_hat = np.array([np.sqrt(q), wrap_angle(np.arctan2(delta[1], delta[0]) - mu_bar[2])])
        delta_x, delta_y = delta

        Fxj = np.zeros((5, len(mu_bar)))
        Fxj[0:3,0:3] = np.eye(3)
        Fxj[3, lm_index] = 1
        Fxj[4, lm_index+1] = 1

        H = (1 / q) * np.array([
            [-np.sqrt(q)*delta_x, -np.sqrt(q)*delta_y, 0,  np.sqrt(q)*delta_x,  np.sqrt(q)*delta_y],
            [delta_y,            -delta_x,            -q, -delta_y,             delta_x]
        ]) @ Fxj

        S = H @ Sigma_bar @ H.T + Q
        K = Sigma_bar @ H.T @ np.linalg.inv(S)

        z_actual = np.array([r, phi])
        dz = z_actual - z_hat
        dz[1] = wrap_angle(dz[1])
        mu_bar = mu_bar + K @ dz
        mu_bar[2] = wrap_angle(mu_bar[2])
        Sigma_bar = (np.eye(len(mu_bar)) - K @ H) @ Sigma_bar

    return mu_bar, Sigma_bar

def run_ekf_slam_test():
    mu, Sigma = initialize_ekf()
    dt = 1.0
    R = np.diag([0.1, 0.1, np.deg2rad(5)]) ** 2
    Q = np.diag([0.5, np.deg2rad(10)]) ** 2

    landmarks = np.array([[5, 10], [10, 0], [15, 15]])

    path = [mu[:2].copy()]
    est_landmarks_over_time = []

    for t in range(10):
        u = [1.0, np.deg2rad(10)] 

        z = []
        c = []
        for j, (lx, ly) in enumerate(landmarks):
            dx = lx - mu[0]
            dy = ly - mu[1]
            r = np.sqrt(dx**2 + dy**2) + np.random.normal(0, np.sqrt(Q[0, 0]))
            phi = wrap_angle(np.arctan2(dy, dx) - mu[2] + np.random.normal(0, np.sqrt(Q[1, 1])))
            z.append([r, phi])
            c.append(j)

        mu, Sigma = ekf_slam_known_correspondences(mu, Sigma, u, z, c, R, Q, dt)
        path.append(mu[:2].copy())
        est_landmarks = np.array(mu[3:]).reshape(-1, 2)
        est_landmarks_over_time.append(est_landmarks)

    return np.array(path), landmarks, est_landmarks_over_time[-1]

path, true_landmarks, estimated_landmarks = run_ekf_slam_test()

plt.figure(figsize=(8, 6))
plt.plot(path[:, 0], path[:, 1], 'b-o', label='Estimated Path')
plt.scatter(true_landmarks[:, 0], true_landmarks[:, 1], c='g', marker='^', label='True Landmarks')
plt.scatter(estimated_landmarks[:, 0], estimated_landmarks[:, 1], c='r', marker='x', label='Estimated Landmarks')
plt.legend()
plt.title('EKF-SLAM with Known Correspondences')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.show()