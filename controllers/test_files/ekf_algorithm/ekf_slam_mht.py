import numpy as np
import matplotlib.pyplot as plt


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def initialize_ekf():
    mu = np.zeros(3)
    Sigma = np.zeros((3, 3))
    return mu, Sigma

def merge_close_landmarks(mu, Sigma, distance_threshold=1.0):
    merged_mu = mu.copy()
    merged_Sigma = Sigma.copy()

    landmarks = merged_mu[3:].reshape(-1, 2)
    N = len(landmarks)

    new_landmarks = []
    keep_indices = []

    used = np.zeros(N, dtype=bool)

    for i in range(N):
        if used[i]:
            continue
        cluster = [landmarks[i]]
        used[i] = True

        for j in range(i + 1, N):
            if used[j]:
                continue
            dist = np.linalg.norm(landmarks[i] - landmarks[j])
            if dist < distance_threshold:
                cluster.append(landmarks[j])
                used[j] = True

        merged = np.mean(cluster, axis=0)
        new_landmarks.append(merged)
        keep_indices.append(i)  

    new_mu = merged_mu[:3]
    for lm in new_landmarks:
        new_mu = np.append(new_mu, lm)

    new_Sigma = np.zeros((len(new_mu), len(new_mu)))
    new_Sigma[:3, :3] = merged_Sigma[:3, :3]
    for i, old_idx in enumerate(keep_indices):
        old_lm_idx = 3 + 2 * old_idx
        new_lm_idx = 3 + 2 * i
        new_Sigma[new_lm_idx:new_lm_idx+2, new_lm_idx:new_lm_idx+2] = \
            merged_Sigma[old_lm_idx:old_lm_idx+2, old_lm_idx:old_lm_idx+2]

    return new_mu, new_Sigma

class Hypothesis:
    def __init__(self, mu, Sigma, history, score=0.0):
        self.mu = mu.copy()
        self.Sigma = Sigma.copy()
        self.history = history[:]
        self.score = score

    def clone(self):
        return Hypothesis(self.mu, self.Sigma, self.history, self.score)


class HypothesisTree:
    def __init__(self, initial_mu, initial_Sigma):
        self.hypotheses = [Hypothesis(initial_mu, initial_Sigma, [])]

    def expand(self, z, R, Q, u, dt, alpha_threshold=9.21):
        new_hypotheses = []

        for hyp in self.hypotheses:
            mu_pred, Sigma_pred = motion_update(hyp.mu, hyp.Sigma, u, R, dt)
            lock_xy = np.isclose(u[0], 0.0) and not np.isclose(u[1], 0.0)

            for i, z_i in enumerate(z):
                associations = self._generate_associations(mu_pred, Sigma_pred, z_i, Q, alpha_threshold)

                if not associations:
                    new_mu, new_Sigma = add_new_landmark(mu_pred, Sigma_pred, z_i)
                    h_new = Hypothesis(new_mu, new_Sigma, hyp.history + [(z_i, None)], hyp.score)
                    new_hypotheses.append(h_new)
                else:
                    for j, dz, H, K, maha in associations:
                        if lock_xy:
                            old_xy = mu_pred[0:2].copy()

                        new_mu = mu_pred + K @ dz
                        new_mu[2] = wrap_angle(new_mu[2])

                        if lock_xy:
                            new_mu[0:2] = old_xy

                        new_Sigma = (np.eye(len(mu_pred)) - K @ H) @ Sigma_pred
                        new_score = hyp.score - maha
                        h_new = Hypothesis(new_mu, new_Sigma, hyp.history + [(z_i, j)], new_score)
                        new_hypotheses.append(h_new)

        if new_hypotheses:
            self.hypotheses = sorted(new_hypotheses, key=lambda h: h.score, reverse=True)[:5]
        else:
            print("No new hypotheses generated. Keeping previous state.")

    def best(self):
        return self.hypotheses[0] if self.hypotheses else None

    def _generate_associations(self, mu, Sigma, z_i, Q, alpha_threshold):
        hypotheses = []
        Nt = (len(mu) - 3) // 2
        for j in range(Nt):
            lm_index = 3 + 2 * j
            delta = mu[lm_index:lm_index + 2] - mu[0:2]
            q = delta.T @ delta
            sqrt_q = np.sqrt(q)
            z_hat = np.array([sqrt_q, wrap_angle(np.arctan2(delta[1], delta[0]) - mu[2])])
            dz = z_i - z_hat
            dz[1] = wrap_angle(dz[1])

            Fxj = np.zeros((5, len(mu)))
            Fxj[0:3, 0:3] = np.eye(3)
            Fxj[3, lm_index] = 1
            Fxj[4, lm_index + 1] = 1

            H = (1 / q) * np.array([
                [-sqrt_q * delta[0], -sqrt_q * delta[1], 0, sqrt_q * delta[0], sqrt_q * delta[1]],
                [delta[1], -delta[0], -q, -delta[1], delta[0]]
            ]) @ Fxj

            S = H @ Sigma @ H.T + Q
            mahalanobis = dz.T @ np.linalg.inv(S) @ dz

            if mahalanobis < alpha_threshold:
                K = Sigma @ H.T @ np.linalg.inv(S)
                hypotheses.append((j, dz, H, K, mahalanobis))

        return hypotheses


def motion_update(mu, Sigma, u, R, dt):
    v, w = u
    theta = mu[2]
    dtheta = w * dt

    if np.isclose(v, 0.0) and not np.isclose(w, 0.0):
        dx, dy = 0.0, 0.0
    elif not np.isclose(w, 0.0):
        dx = -v / w * np.sin(theta) + v / w * np.sin(theta + dtheta)
        dy = v / w * np.cos(theta) - v / w * np.cos(theta + dtheta)
    else:
        dx = v * dt * np.cos(theta)
        dy = v * dt * np.sin(theta)

    mu_bar = mu.copy()
    mu_bar[0] += dx
    mu_bar[1] += dy
    mu_bar[2] = wrap_angle(mu_bar[2] + dtheta)

    n_landmarks = (len(mu) - 3) // 2
    Fx = np.hstack([np.eye(3), np.zeros((3, 2 * n_landmarks))])

    if np.isclose(v, 0.0) and not np.isclose(w, 0.0):
        Gx = np.zeros((3, 3))
    elif not np.isclose(w, 0.0):
        Gx = np.array([
            [0, 0, v / w * np.cos(theta) - v / w * np.cos(theta + dtheta)],
            [0, 0, v / w * np.sin(theta) - v / w * np.sin(theta + dtheta)],
            [0, 0, 0]
        ])
    else:
        Gx = np.array([
            [0, 0, -v * dt * np.sin(theta)],
            [0, 0, v * dt * np.cos(theta)],
            [0, 0, 0]
        ])

    G = np.eye(len(mu)) + Fx.T @ Gx @ Fx
    Sigma_bar = G @ Sigma @ G.T + Fx.T @ R @ Fx

    return mu_bar, Sigma_bar


def add_new_landmark(mu, Sigma, z_i):
    r, phi = z_i
    lx = mu[0] + r * np.cos(phi + mu[2])
    ly = mu[1] + r * np.sin(phi + mu[2])
    mu_new = np.append(mu, [lx, ly])
    new_Sigma = np.zeros((len(mu_new), len(mu_new)))
    new_Sigma[:Sigma.shape[0], :Sigma.shape[1]] = Sigma
    new_Sigma[-2:, -2:] = np.diag([4.0, 4.0])
    return mu_new, new_Sigma


def run_ekf_slam_mht():
    mu, Sigma = initialize_ekf()
    dt = 1.0
    R = np.diag([0.1, 0.1, np.deg2rad(5)]) ** 2
    Q = np.diag([0.5, np.deg2rad(2.0)]) ** 2

    landmarks = np.array([[5, 10], [10, 0], [15, 15]])
    path = [mu[:2].copy()]
    est_landmarks_over_time = []

    tree = HypothesisTree(mu, Sigma)

    for t in range(50):
        if t < 10:
            u = [1.0, np.deg2rad(10)]
        elif 10 <= t < 28:
            u = [0.0, np.deg2rad(10)]
        else:
            u = [1.0, np.deg2rad(10)]
        
        z = []
        best_hyp = tree.best()
        if best_hyp is None:
            print("No valid hypotheses remain. Using last known state.")
            best_hyp = Hypothesis(mu, Sigma, [])

        for (lx, ly) in landmarks:
            dx = lx - best_hyp.mu[0]
            dy = ly - best_hyp.mu[1]
            r = np.sqrt(dx ** 2 + dy ** 2) + np.random.normal(0, np.sqrt(Q[0, 0]))
            phi = wrap_angle(np.arctan2(dy, dx) - best_hyp.mu[2] + np.random.normal(0, np.sqrt(Q[1, 1])))
            z.append([r, phi])

        tree.expand(z, R, Q, u, dt)
        best_hyp = tree.best()
        mu, Sigma = best_hyp.mu, best_hyp.Sigma
        mu, Sigma = merge_close_landmarks(mu, Sigma, distance_threshold=3.0)
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
    plt.title('EKF-SLAM with MHT Association')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


run_ekf_slam_mht()