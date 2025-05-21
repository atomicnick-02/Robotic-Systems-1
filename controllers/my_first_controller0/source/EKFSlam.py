import numpy as np
import matplotlib.pyplot as plt

class EKF_SLAM:
    """
    Extended Kalman Filter SLAM with multi-hypothesis data association.

    Attributes:
        hypotheses        List of Hypothesis instances (each has mu, Sigma, score).
        R                 Odometry noise covariance (3×3).
        Q                 Sensor noise covariance for range-bearing (2×2).
        dt                Time step for control integration.
        alpha_threshold   Chi-squared threshold for Mahalanobis gating.
        min_landmark_distance  Merge threshold: landmarks closer than this are merged.
        max_hypotheses    Maximum number of hypotheses to maintain.
    """

    class Hypothesis:
        """Stores a single SLAM hypothesis: state mean, covariance, and score."""
        def __init__(self, mu: np.ndarray, Sigma: np.ndarray, score: float = 0.0):
            self.mu = mu.copy()
            self.Sigma = Sigma.copy()
            self.score = score

        def clone(self):
            """Return a deep copy of this hypothesis."""
            return EKF_SLAM.Hypothesis(self.mu, self.Sigma, self.score)

    def __init__(self,
                 R: np.ndarray = np.diag([0.1, 0.1, np.deg2rad(5)])**2,
                 Q: np.ndarray = np.diag([1.0, np.deg2rad(5)])**2,
                 dt: float = 0.010,
                 alpha_threshold: float = 9.21,
                 min_landmark_distance: float = 0.5,
                 max_hypotheses: int = 5):
        # Initialize with a single zeroed pose (no landmarks yet)
        init_mu = np.zeros(3)              # [x, y, θ]
        init_Sigma = np.eye(3)             # Covariance of pose only
        self.hypotheses = [self.Hypothesis(init_mu, init_Sigma)]

        # Noise parameters
        self.R = R                          # Motion (odometry) noise
        self.Q = Q                          # Sensor (range-bearing) noise

        # Filter settings
        self.dt = dt
        self.alpha_threshold = alpha_threshold
        self.min_landmark_distance = min_landmark_distance
        self.max_hypotheses = max_hypotheses

    @staticmethod
    def wrap_angle(angle: float) -> float:
        """
        Normalize angle into [-π, π).
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # ---- Public setters ----
    def set_noise_parameters(self, R: np.ndarray = None, Q: np.ndarray = None):
        """Optionally update motion (R) and/or sensor (Q) noise covariances."""
        if R is not None:
            self.R = R
        if Q is not None:
            self.Q = Q

    def set_time_step(self, dt: float):
        """Set the control integration timestep (s)."""
        self.dt = dt

    def set_landmark_threshold(self, alpha_threshold: float, min_distance: float = None):
        """
        Adjust data-association and landmark-merge thresholds.
        :param alpha_threshold: χ² threshold for gating
        :param min_distance: minimum distance to merge landmarks (meters)
        """
        self.alpha_threshold = alpha_threshold
        if min_distance is not None:
            self.min_landmark_distance = min_distance

    # ---- Public getters ----
    def get_state(self):
        """
        Retrieve the best hypothesis’s pose and landmarks.
        :return: (pose: [x, y, θ], landmarks: Nx2 array or None)
        """
        best = self.hypotheses[0]
        pose = best.mu[:3].copy()
        
        if best.mu.size > 3:
            lms = best.mu[3:].reshape(-1, 2).copy()
        else:
            lms = None
        return pose, lms

    def get_covariance(self) -> np.ndarray:
        """Return the covariance of the best hypothesis."""
        return self.hypotheses[0].Sigma.copy()

    # ---- Core EKF-SLAM steps ----
    def predict(self, u: np.ndarray):
        """
        Perform the motion (prediction) update on all hypotheses.
        :param u: control input [v, ω]
        """
        updated = []
        for hyp in self.hypotheses:
            mu_bar, Sigma_bar = self._motion_update(hyp.mu, hyp.Sigma, u)
            updated.append(self.Hypothesis(mu_bar, Sigma_bar, hyp.score))
        self.hypotheses = updated

    def correct(self, measurements: np.ndarray):
        """
        Perform the measurement (correction) update for each observation.
        :param measurements: list/array of [range, bearing] observations
        """
        new_hyps = []
        for hyp in self.hypotheses:
            # For each individual measurement
            for z in measurements:
                associations = self._associate(hyp.mu, hyp.Sigma, z)

                # If no existing landmarks match, initialize a new one
                if not associations:
                    mu_new, Sigma_new = self._add_new_landmark(hyp.mu, hyp.Sigma, z)
                    new_hyps.append(self.Hypothesis(mu_new, Sigma_new, hyp.score))
                else:
                    # Branch on every valid association
                    for (_, dz, H, K, maha) in associations:
                        mu_new = hyp.mu + K @ dz
                        mu_new[2] = self.wrap_angle(mu_new[2])  # normalize heading
                        Sigma_new = (np.eye(len(mu_new)) - K @ H) @ hyp.Sigma
                        score_new = hyp.score - maha           # lower Mahalanobis → higher score
                        new_hyps.append(self.Hypothesis(mu_new, Sigma_new, score_new))

        # Keep only the top-scoring hypotheses
        self.hypotheses = sorted(new_hyps, key=lambda h: h.score, reverse=True)[:self.max_hypotheses]

        # Merge any nearby landmarks in the best hypothesis
        best = self.hypotheses[0]
        best.mu, best.Sigma = self._merge_close_landmarks(
            best.mu, best.Sigma, self.min_landmark_distance
        )

    # ---- Visualization ----
    def plot_landmarks(self):
        """
        Scatter-plot of landmarks and current robot pose (arrow).
        Useful for live SLAM display in a loop.
        """
        pose, lms = self.get_state()
        plt.cla()
        if lms is not None:
            plt.scatter(lms[:, 0], lms[:, 1], c='r', label='Landmarks')
        # Robot pose as a unit arrow
        plt.quiver(
            pose[0], pose[1],
            np.cos(pose[2]), np.sin(pose[2]),
            angles='xy', scale_units='xy', scale=1,
            color='b', label='Robot'
        )
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('EKF SLAM Map')
        plt.grid(True)
        plt.legend()
        plt.pause(0.001)

    # ---- Internal helper methods ----
    def _motion_update(self, mu: np.ndarray, Sigma: np.ndarray, u: np.ndarray):
        """
        EKF motion update (prediction step) using the unicycle model.
        :param mu:   current state [x, y, θ, (landmarks...)]
        :param Sigma: current covariance
        :param u:    control [v, ω]
        :returns: (mu_bar, Sigma_bar)
        """
        v, w = u
        θ = mu[2]
        dθ = w * self.dt

        # Compute pose increment
        if np.isclose(w, 0.0):
            dx = v * self.dt * np.cos(θ)
            dy = v * self.dt * np.sin(θ)
        else:
            dx = -v / w * np.sin(θ) + v / w * np.sin(θ + dθ)
            dy =  v / w * np.cos(θ) - v / w * np.cos(θ + dθ)

        # Propagate mean
        mu_bar = mu.copy()
        mu_bar[0] += dx
        mu_bar[1] += dy
        mu_bar[2] = self.wrap_angle(θ + dθ)

        # Jacobians
        n_lm = (len(mu) - 3) // 2
        Fx = np.hstack([np.eye(3), np.zeros((3, 2*n_lm))])  # picks out the pose block

        # Pose‐only Jacobian Gx
        if np.isclose(w, 0.0):
            Gx = np.array([
                [0, 0, -v*self.dt*np.sin(θ)],
                [0, 0,  v*self.dt*np.cos(θ)],
                [0, 0, 0]
            ])
        else:
            Gx = np.array([
                [0, 0, (v/w)*(np.cos(θ) - np.cos(θ + dθ))],
                [0, 0, (v/w)*(np.sin(θ) - np.sin(θ + dθ))],
                [0, 0, 0]
            ])

        # Full Jacobian of the state transition
        G = np.eye(len(mu)) + Fx.T @ Gx @ Fx

        # Covariance propagation
        Sigma_bar = G @ Sigma @ G.T + Fx.T @ self.R @ Fx
        return mu_bar, Sigma_bar

    def _associate(self, mu: np.ndarray, Sigma: np.ndarray, z: np.ndarray):
        """
        Mahalanobis‐gate existing landmarks against measurement z.
        :returns: list of valid associations [(idx, dz, H, K, mahalanobis_dist), ...]
        """
        associations = []
        num_lm = (len(mu) - 3) // 2
        # print(f"z: {z}")
        for j in range(num_lm):
            idx = 3 + 2*j
            delta = mu[idx:idx+2] - mu[:2]
            q = delta @ delta
            r_pred = np.sqrt(q)
            φ_pred = self.wrap_angle(np.arctan2(delta[1], delta[0]) - mu[2])

            # Innovation
            dz = z - np.array([r_pred, φ_pred])
            dz[1] = self.wrap_angle(dz[1])

            # Measurement Jacobian
            Fxj = np.zeros((5, len(mu)))
            Fxj[:3, :3] = np.eye(3)
            Fxj[3, idx]     = 1
            Fxj[4, idx + 1] = 1

            H_small = (1/q) * np.array([
                [-r_pred*delta[0], -r_pred*delta[1], 0,  r_pred*delta[0],  r_pred*delta[1]],
                [ delta[1],       -delta[0],        -q, -delta[1],         delta[0]]
            ])
            H = H_small @ Fxj

            # Innovation covariance and Mahalanobis distance
            S = H @ Sigma @ H.T + self.Q
            maha = float(dz.T @ np.linalg.inv(S) @ dz)

            if maha < self.alpha_threshold:
                K = Sigma @ H.T @ np.linalg.inv(S)
                associations.append((j, dz, H, K, maha))

        return associations

    def _add_new_landmark(self, mu: np.ndarray, Sigma: np.ndarray, z: np.ndarray):
        """
        Initialize a new landmark in the map from measurement z = [r, φ].
        Returns augmented (mu, Sigma).
        """
        r, φ = z
        # Landmark position in world frame
        lx = mu[0] + r * np.cos(φ + mu[2])
        ly = mu[1] + r * np.sin(φ + mu[2])

        # Augment state vector
        mu_new = np.hstack([mu, [lx, ly]])

        # Augment covariance: 
        #   existing covariances remain, 
        #   new landmark covariance initialized large (1 m²)
        n = len(mu_new)
        Sigma_new = np.zeros((n, n))
        Sigma_new[:len(Sigma), :len(Sigma)] = Sigma
        Sigma_new[-2:, -2:] = np.diag([1.0, 1.0])
        return mu_new, Sigma_new

    def _merge_close_landmarks(self, mu: np.ndarray, Sigma: np.ndarray, threshold: float):
        """
        Merge landmarks whose Euclidean distance is below threshold.
        Keeps the first of each cluster, averages positions, carries over covariance.
        """
        if mu.size <= 3:
            return mu, Sigma

        lms = mu[3:].reshape(-1, 2)
        N = lms.shape[0]
        used = np.zeros(N, bool)
        merged_lms = []
        kept_indices = []

        # Cluster by proximity
        for i in range(N):
            if used[i]:
                continue
            cluster = [lms[i]]
            used[i] = True
            for j in range(i+1, N):
                if (not used[j]) and np.linalg.norm(lms[i] - lms[j]) < threshold:
                    cluster.append(lms[j])
                    used[j] = True
            merged_lm = np.mean(cluster, axis=0)
            merged_lms.append(merged_lm)
            kept_indices.append(i)

        # Build new state vector and covariance
        new_mu = np.hstack([mu[:3], np.vstack(merged_lms).ravel()])
        m = new_mu.size
        new_Sigma = np.zeros((m, m))

        # Copy pose covariance
        new_Sigma[:3, :3] = Sigma[:3, :3]

        # Copy landmark covariances
        for k, old_i in enumerate(kept_indices):
            old_idx = 3 + 2*old_i
            new_idx = 3 + 2*k
            new_Sigma[new_idx:new_idx+2, new_idx:new_idx+2] = Sigma[old_idx:old_idx+2, old_idx:old_idx+2]

        return new_mu, new_Sigma
