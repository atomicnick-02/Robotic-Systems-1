import numpy as np
import matplotlib.pyplot as plt

# Helper functions
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def initialize_ekf():
    mu = np.zeros(3) 
    Sigma = np.zeros((3,3)) 
    return mu, Sigma

# Main EKF SLAM with unknown correspondences
def ekf_slam_unknown_correspondences(mu, Sigma, u, z, R, Q, dt, alpha_threshold=9.21):
    n_landmarks = (len(mu) - 3) // 2
    Fx = np.hstack([np.eye(3), np.zeros((3, 2 * n_landmarks))])

    v, w = u
    theta = mu[2]
    dtheta = w * dt
    if abs(w) > 1e-6:
        dx = -v / w * np.sin(theta) + v / w * np.sin(theta + dtheta)
        dy = v / w * np.cos(theta) - v / w * np.cos(theta + dtheta)
    else:
        dx = v * dt * np.cos(theta)
        dy = v * dt * np.sin(theta)

    mu_bar = mu.copy()
    mu_bar[0] += dx
    mu_bar[1] += dy
    mu_bar[2] = wrap_angle(mu_bar[2] + dtheta)

    Gx = np.array([
        [0, 0, v/w * np.cos(theta) - v/w * np.cos(theta + dtheta)],
        [0, 0, v/w * np.sin(theta) - v/w * np.sin(theta + dtheta)],
        [0, 0, 0]
    ]) if abs(w) > 1e-6 else np.array([
        [0, 0, -v * dt * np.sin(theta)],
        [0, 0, -v * dt * np.cos(theta)],
        [0, 0, 0]
    ])

    G = np.eye(len(mu)) + Fx.T @ Gx @ Fx
    Sigma_bar = G @ Sigma @ G.T + Fx.T @ R @ Fx

    for i in range(len(z)):
        r_i, phi_i = z[i]
        z_i = np.array([r_i, phi_i])

        min_mahalanobis = float('inf')
        best_dz = None
        best_H = None
        best_K = None

        Nt = (len(mu_bar) - 3) // 2
        for j in range(Nt):
            lm_index = 3 + 2 * j
            delta = mu_bar[lm_index:lm_index+2] - mu_bar[0:2]
            q = delta.T @ delta
            sqrt_q = np.sqrt(q)
            z_hat = np.array([
                sqrt_q,
                wrap_angle(np.arctan2(delta[1], delta[0]) - mu_bar[2])
            ])
            dz = z_i - z_hat
            dz[1] = wrap_angle(dz[1])

            Fxj = np.zeros((5, len(mu_bar)))
            Fxj[0:3, 0:3] = np.eye(3)
            Fxj[3, lm_index] = 1
            Fxj[4, lm_index+1] = 1

            H = (1 / q) * np.array([
                [-sqrt_q * delta[0], -sqrt_q * delta[1], 0, sqrt_q * delta[0], sqrt_q * delta[1]],
                [delta[1], -delta[0], -q, -delta[1], delta[0]]
            ]) @ Fxj

            S = H @ Sigma_bar @ H.T + Q
            mahalanobis = dz.T @ np.linalg.inv(S) @ dz

            if mahalanobis < min_mahalanobis:
                min_mahalanobis = mahalanobis
                best_dz = dz
                best_H = H
                best_K = Sigma_bar @ H.T @ np.linalg.inv(S)

        if min_mahalanobis > alpha_threshold:
            # New landmark
            lx = mu_bar[0] + r_i * np.cos(phi_i + mu_bar[2])
            ly = mu_bar[1] + r_i * np.sin(phi_i + mu_bar[2])

            # Only add if not too close to an existing landmark
            too_close = False
            for j in range((len(mu_bar) - 3) // 2):
                idx = 3 + 2*j
                existing_lx = mu_bar[idx]
                existing_ly = mu_bar[idx+1]
                dist = np.hypot(existing_lx - lx, existing_ly - ly)
                if dist < 10:  # Threshold distance for "already seen"
                    too_close = True
                    break

            if not too_close:
                mu_bar = np.append(mu_bar, [lx, ly])
                new_Sigma = np.zeros((len(mu_bar), len(mu_bar)))
                new_Sigma[:Sigma_bar.shape[0], :Sigma_bar.shape[1]] = Sigma_bar
                new_Sigma[-2:, -2:] = np.diag([1, 1])  
                Sigma_bar = new_Sigma
                print(f"✅ New landmark added at ({lx:.2f}, {ly:.2f}) with Mahalanobis distance {min_mahalanobis:.2f}")
            else:
                print(f"⚠️ Skipped adding landmark at ({lx:.2f}, {ly:.2f}) — too close to existing one.")

        else:
            mu_bar = mu_bar + best_K @ best_dz
            mu_bar[2] = wrap_angle(mu_bar[2])
            Sigma_bar = (np.eye(len(mu_bar)) - best_K @ best_H) @ Sigma_bar

    return mu_bar, Sigma_bar


def run_ekf_slam_test_unknown_correspondences():
    mu, Sigma = initialize_ekf()
    dt = 1.0
    R = np.diag([0.1, 0.1, np.deg2rad(5)]) ** 2
    Q = np.diag([0.5, np.deg2rad(2.0)]) ** 2

    landmarks = np.array([[5, 10], [10, 0], [15, 15]])
    path = [mu[:2].copy()]
    est_landmarks_over_time = []

    for t in range(20):
        u = [1.0, np.deg2rad(10)]
        z = []
        for (lx, ly) in landmarks:
            dx = lx - mu[0]
            dy = ly - mu[1]
            r = np.sqrt(dx**2 + dy**2) + np.random.normal(0, np.sqrt(Q[0, 0]))
            phi = wrap_angle(np.arctan2(dy, dx) - mu[2] + np.random.normal(0, np.sqrt(Q[1, 1])))
            z.append([r, phi])

        mu, Sigma = ekf_slam_unknown_correspondences(mu, Sigma, u, z, R, Q, dt)
        path.append(mu[:2].copy())
        est_landmarks = np.array(mu[3:]).reshape(-1, 2)
        est_landmarks_over_time.append(est_landmarks)

    path = np.array(path)
    final_landmarks = est_landmarks_over_time[-1]

    plt.figure(figsize=(8, 6))
    plt.plot(path[:, 0], path[:, 1], 'b-o', label='Estimated Path')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='g', marker='^', label='True Landmarks')
    if len(final_landmarks) > 0:
        plt.scatter(final_landmarks[:, 0], final_landmarks[:, 1], c='r', marker='x', label='Estimated Landmarks')
    plt.legend()
    plt.title('EKF-SLAM with Unknown Correspondences')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

run_ekf_slam_test_unknown_correspondences()
