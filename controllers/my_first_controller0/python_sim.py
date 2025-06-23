"""
Extended Kalman Filter SLAM with Multiple Hypothesis Tracking (EKF-SLAM-MHT)

This implementation provides a robust SLAM solution that maintains multiple hypotheses
for data association uncertainty, allowing the algorithm to handle ambiguous landmark
associations more effectively than single-hypothesis approaches.

Author: Cleaned and commented version
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import distance_matrix


class EKF_SLAM:
    """
    Extended Kalman Filter SLAM with Multiple Hypothesis Tracking
    
    This class implements SLAM using EKF with multiple hypotheses to handle
    data association uncertainty. It maintains several possible state estimates
    and selects the best one based on likelihood scores.
    """
    
    def __init__(self):
        """Initialize EKF-SLAM with default parameters"""
        # Start with single hypothesis: robot at origin with no landmarks
        self.hypotheses = [self.Hypothesis(np.zeros(3), np.eye(3))]
        
        # Motion noise covariance (x, y, theta)
        self.R = np.diag([0.1, 0.1, np.deg2rad(5)]) ** 2
        
        # Observation noise covariance (range, bearing)
        self.Q = np.diag([1.0, np.deg2rad(5)]) ** 2
        
        # Time step for motion model
        self.dt = 0.032
        
        # Chi-squared threshold for data association (95% confidence)
        self.alpha_threshold = 9.21
        
        # Minimum distance to merge close landmarks
        self.min_landmark_distance = 4
        
        # Maximum number of hypotheses to maintain
        self.max_hypotheses = 5

    class Hypothesis:
        """
        Single hypothesis containing state estimate and covariance
        
        State format: [x, y, theta, lm1_x, lm1_y, lm2_x, lm2_y, ...]
        where (x,y,theta) is robot pose and (lmi_x, lmi_y) are landmark positions
        """
        
        def __init__(self, mu, Sigma, score=0.0):
            """
            Initialize hypothesis
            
            Args:
                mu: State mean vector
                Sigma: State covariance matrix  
                score: Log-likelihood score (higher is better)
            """
            self.mu = mu.copy()
            self.Sigma = Sigma.copy()
            self.score = score

        def clone(self):
            """Create deep copy of hypothesis"""
            return EKF_SLAM.Hypothesis(self.mu.copy(), self.Sigma.copy(), self.score)

    @staticmethod
    def wrap_angle(angle):
        """Wrap angle to [-pi, pi] range"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def set_noise_parameters(self, R=None, Q=None):
        """
        Set noise parameters
        
        Args:
            R: Motion noise covariance matrix (3x3)
            Q: Observation noise covariance matrix (2x2)
        """
        if R is not None: 
            self.R = R
        if Q is not None: 
            self.Q = Q

    def set_time_step(self, dt):
        """Set time step for motion model"""
        self.dt = dt

    def set_landmark_threshold(self, alpha_threshold, min_distance=None):
        """
        Set landmark association and merging parameters
        
        Args:
            alpha_threshold: Chi-squared threshold for data association
            min_distance: Minimum distance to merge close landmarks
        """
        self.alpha_threshold = alpha_threshold
        if min_distance is not None:
            self.min_landmark_distance = min_distance

    def get_state(self):
        """
        Get current best state estimate
        
        Returns:
            pose: Robot pose [x, y, theta]
            landmarks: Landmark positions as Nx2 array (or None if no landmarks)
        """
        best = self.hypotheses[0]
        pose = best.mu[:3].copy()
        landmarks = best.mu[3:].reshape(-1, 2).copy() if len(best.mu) > 3 else None
        return pose, landmarks

    def get_covariance(self):
        """Get current best covariance estimate"""
        return self.hypotheses[0].Sigma.copy()

    def predict(self, u):
        """
        Prediction step: propagate all hypotheses through motion model
        
        Args:
            u: Control input [linear_velocity, angular_velocity]
        """
        new_hypotheses = []
        for hyp in self.hypotheses:
            mu_bar, Sigma_bar = self._motion_update(hyp.mu, hyp.Sigma, u)
            new_hypotheses.append(self.Hypothesis(mu_bar, Sigma_bar, hyp.score))
        self.hypotheses = new_hypotheses

    def update(self, u, z):
        """
        Full EKF update: prediction + correction
        
        Args:
            u: Control input [linear_velocity, angular_velocity]
            z: List of observations [[range, bearing], ...]
        """
        self.predict(u)
        if z:
            self.correct(z)

    def correct(self, z):
        """
        Correction step: update hypotheses with observations
        Creates new hypotheses for each possible data association
        
        Args:
            z: List of observations [[range, bearing], ...]
        """
        new_hypotheses = []
        
        # For each existing hypothesis
        for hyp in self.hypotheses:
            # For each observation
            for zi in z:
                # Find possible landmark associations
                associations = self._associate(hyp.mu, hyp.Sigma, zi)
                
                if not associations:
                    # No association found - add as new landmark
                    mu_new, Sigma_new = self._add_new_landmark(hyp.mu, hyp.Sigma, zi)
                    new_hypotheses.append(self.Hypothesis(mu_new, Sigma_new, hyp.score))
                else:
                    # Create new hypothesis for each possible association
                    for j, dz, H, K, maha_dist in associations:
                        # Standard EKF update
                        mu_new = hyp.mu + K @ dz
                        mu_new[2] = self.wrap_angle(mu_new[2])  # Wrap robot heading
                        Sigma_new = (np.eye(len(mu_new)) - K @ H) @ hyp.Sigma
                        
                        # Update score (subtract Mahalanobis distance as penalty)
                        score = hyp.score - maha_dist
                        new_hypotheses.append(self.Hypothesis(mu_new, Sigma_new, score))

        # Keep only the best hypotheses
        self.hypotheses = sorted(new_hypotheses, key=lambda h: h.score, reverse=True)[:self.max_hypotheses]
        
        # Merge close landmarks in best hypothesis to prevent map fragmentation
        best = self.hypotheses[0]
        best.mu, best.Sigma = self._merge_close_landmarks(best.mu, best.Sigma, self.min_landmark_distance)

    def _motion_update(self, mu, Sigma, u):
        """
        Motion model update using bicycle/unicycle model
        
        Args:
            mu: Current state mean
            Sigma: Current state covariance
            u: Control input [v, w] (linear velocity, angular velocity)
            
        Returns:
            mu_bar: Predicted state mean
            Sigma_bar: Predicted state covariance
        """
        v, w = u
        theta = mu[2]
        dtheta = w * self.dt

        # Predict robot motion using bicycle model
        if np.isclose(w, 0.0):
            # Straight line motion
            dx = v * self.dt * np.cos(theta)
            dy = v * self.dt * np.sin(theta)
        else:
            # Circular arc motion
            dx = -v / w * np.sin(theta) + v / w * np.sin(theta + dtheta)
            dy = v / w * np.cos(theta) - v / w * np.cos(theta + dtheta)

        # Update robot pose
        mu_bar = mu.copy()
        mu_bar[0] += dx
        mu_bar[1] += dy
        mu_bar[2] = self.wrap_angle(mu_bar[2] + dtheta)

        # Jacobians for covariance propagation
        n_landmarks = (len(mu) - 3) // 2
        Fx = np.hstack([np.eye(3), np.zeros((3, 2 * n_landmarks))])  # Select robot state

        if np.isclose(w, 0.0):
            # Jacobian for straight motion
            Gx = np.array([
                [0, 0, -v * self.dt * np.sin(theta)],
                [0, 0, v * self.dt * np.cos(theta)],
                [0, 0, 0]
            ])
        else:
            # Jacobian for circular motion
            Gx = np.array([
                [0, 0, v / w * np.cos(theta) - v / w * np.cos(theta + dtheta)],
                [0, 0, v / w * np.sin(theta) - v / w * np.sin(theta + dtheta)],
                [0, 0, 0]
            ])

        # Full state Jacobian (landmarks don't move)
        G = np.eye(len(mu)) + Fx.T @ Gx @ Fx
        
        # Propagate covariance
        Sigma_bar = G @ Sigma @ G.T + Fx.T @ self.R @ Fx
        
        return mu_bar, Sigma_bar

    def _associate(self, mu, Sigma, z):
        """
        Data association: find landmarks that could have generated observation z
        
        Args:
            mu: Current state mean
            Sigma: Current state covariance
            z: Observation [range, bearing]
            
        Returns:
            List of valid associations: [(landmark_idx, innovation, H, K, mahalanobis_dist), ...]
        """
        associations = []
        n_landmarks = (len(mu) - 3) // 2
        
        # Check association with each existing landmark
        for j in range(n_landmarks):
            landmark_idx = 3 + 2 * j
            
            # Expected observation from this landmark
            delta = mu[landmark_idx:landmark_idx + 2] - mu[0:2]  # Landmark relative to robot
            q = delta.T @ delta  # Squared distance
            sqrt_q = np.sqrt(q)
            
            # Predicted observation
            z_hat = np.array([
                sqrt_q,  # Range
                self.wrap_angle(np.arctan2(delta[1], delta[0]) - mu[2])  # Bearing
            ])
            
            # Innovation (observation - prediction)
            dz = z - z_hat
            dz[1] = self.wrap_angle(dz[1])  # Wrap bearing difference

            # Observation Jacobian
            # Select relevant state variables: [x, y, theta, landmark_x, landmark_y]
            Fxj = np.zeros((5, len(mu)))
            Fxj[0:3, 0:3] = np.eye(3)           # Robot pose
            Fxj[3, landmark_idx] = 1             # Landmark x
            Fxj[4, landmark_idx + 1] = 1         # Landmark y

            # Jacobian of observation model w.r.t. selected states
            H = (1 / q) * np.array([
                [-sqrt_q * delta[0], -sqrt_q * delta[1], 0, sqrt_q * delta[0], sqrt_q * delta[1]],
                [delta[1], -delta[0], -q, -delta[1], delta[0]]
            ]) @ Fxj

            # Innovation covariance
            S = H @ Sigma @ H.T + self.Q
            
            # Mahalanobis distance for gating
            maha_dist = dz.T @ np.linalg.inv(S) @ dz

            # Check if association is valid (within gate)
            if maha_dist < self.alpha_threshold:
                K = Sigma @ H.T @ np.linalg.inv(S)  # Kalman gain
                associations.append((j, dz, H, K, maha_dist))

        return associations

    def _add_new_landmark(self, mu, Sigma, z):
        """
        Add new landmark to map based on observation
        
        Args:
            mu: Current state mean
            Sigma: Current state covariance  
            z: Observation [range, bearing]
            
        Returns:
            mu_new: Augmented state mean
            Sigma_new: Augmented state covariance
        """
        r, phi = z
        
        # Compute landmark position in global frame
        lx = mu[0] + r * np.cos(phi + mu[2])
        ly = mu[1] + r * np.sin(phi + mu[2])
        
        # Augment state
        mu_new = np.append(mu, [lx, ly])
        
        # Augment covariance (initialize landmark with high uncertainty)
        Sigma_new = np.zeros((len(mu_new), len(mu_new)))
        Sigma_new[:len(Sigma), :len(Sigma)] = Sigma
        Sigma_new[-2:, -2:] = np.diag([5.0, 5.0])  # High initial uncertainty
        
        return mu_new, Sigma_new

    def _merge_close_landmarks(self, mu, Sigma, distance_threshold):
        """
        Merge landmarks that are too close together to prevent map fragmentation
        
        Args:
            mu: Current state mean
            Sigma: Current state covariance
            distance_threshold: Minimum allowed distance between landmarks
            
        Returns:
            mu_merged: State with merged landmarks
            Sigma_merged: Covariance with merged landmarks
        """
        if len(mu) <= 3:  # No landmarks to merge
            return mu, Sigma

        landmarks = mu[3:].reshape(-1, 2)
        n_landmarks = landmarks.shape[0]
        used = np.zeros(n_landmarks, dtype=bool)
        new_landmarks = []
        keep_indices = []

        # Group nearby landmarks
        for i in range(n_landmarks):
            if used[i]:
                continue
                
            cluster = [landmarks[i]]
            cov_i = Sigma[3 + 2*i:3 + 2*i + 2, 3 + 2*i:3 + 2*i + 2]
            used[i] = True

            # Find landmarks close to landmark i
            for j in range(i + 1, n_landmarks):
                if used[j]:
                    continue
                    
                cov_j = Sigma[3 + 2*j:3 + 2*j + 2, 3 + 2*j:3 + 2*j + 2]
                diff = landmarks[i] - landmarks[j]
                cov_sum = cov_i + cov_j
                
                try:
                    # Use Mahalanobis distance for merging decision
                    maha_dist = diff.T @ np.linalg.inv(cov_sum) @ diff
                except np.linalg.LinAlgError:
                    continue  # Skip if covariance is singular

                if maha_dist < distance_threshold:
                    cluster.append(landmarks[j])
                    used[j] = True

            # Merge cluster by averaging positions
            merged_landmark = np.mean(cluster, axis=0)
            new_landmarks.append(merged_landmark)
            keep_indices.append(i)

        # Reconstruct state and covariance
        mu_new = mu[:3]  # Keep robot pose
        for landmark in new_landmarks:
            mu_new = np.append(mu_new, landmark)

        # Reconstruct covariance matrix
        Sigma_new = np.zeros((len(mu_new), len(mu_new)))
        Sigma_new[:3, :3] = Sigma[:3, :3]  # Keep robot covariance
        
        for i, old_idx in enumerate(keep_indices):
            old_lm_idx = 3 + 2 * old_idx
            new_lm_idx = 3 + 2 * i
            Sigma_new[new_lm_idx:new_lm_idx + 2, new_lm_idx:new_lm_idx + 2] = \
                Sigma[old_lm_idx:old_lm_idx + 2, old_lm_idx:old_lm_idx + 2]

        return mu_new, Sigma_new

    def plot_landmarks(self):
        """Plot current map estimate with uncertainty ellipses"""
        pose, landmarks = self.get_state()
        Sigma = self.get_covariance()
        plt.cla()
        ax = plt.gca()

        # Plot landmarks with uncertainty ellipses
        if landmarks is not None:
            plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='o', 
                       s=50, label='Estimated Landmarks')
            
            for i in range(landmarks.shape[0]):
                lm_idx = 3 + 2 * i
                cov = Sigma[lm_idx:lm_idx+2, lm_idx:lm_idx+2]
                mean = landmarks[i]
                
                # Compute uncertainty ellipse
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                width, height = 2 * np.sqrt(9.21 * eigenvals)  # 99% confidence
                
                ellipse = patches.Ellipse(xy=mean, width=width, height=height, 
                                        angle=angle, edgecolor='k', facecolor='none', 
                                        linewidth=1, alpha=0.7)
                ax.add_patch(ellipse)

        # Plot robot pose
        plt.quiver(pose[0], pose[1], np.cos(pose[2]), np.sin(pose[2]),
                   angles='xy', scale_units='xy', scale=0.5,
                   color='b', width=0.005, label='Robot Pose')

        plt.xlim(-20, 25)
        plt.ylim(-20, 25)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('EKF-SLAM Map Estimate')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.pause(0.001)


def wrap_angle(angle):
    """Utility function to wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def run_ekf_slam_simulation():
    """
    Run EKF-SLAM simulation with autonomous exploration
    
    The robot explores a rectangular area containing landmarks using a
    sweeping pattern with obstacle avoidance.
    """
    # Initialize EKF-SLAM
    ekf_slam = EKF_SLAM()
    ekf_slam.set_time_step(1.0)
    ekf_slam.set_noise_parameters(
        R=np.diag([0.1, 0.1, np.deg2rad(5)]) ** 2,  # Motion noise
        Q=np.diag([0.5, np.deg2rad(1.0)]) ** 2       # Observation noise
    )
    ekf_slam.set_landmark_threshold(alpha_threshold=9.21, min_distance=4)

    # Define landmark positions (ground truth)
    landmarks = np.array([
        [5, 10], [10, 0], [15, 15], [2, 2], [3, 8], 
        [10, 10], [12, 4], [7, 4], [16, 10]
    ])

    # Define exploration area based on landmark distribution
    margin = 5.0
    x_min, y_min = np.min(landmarks, axis=0)
    x_max, y_max = np.max(landmarks, axis=0)
    box_limits = [x_min - margin, x_max + margin, y_min - margin, y_max + margin]

    # Compute sweep spacing based on landmark density
    if len(landmarks) > 1:
        dists = distance_matrix(landmarks, landmarks)
        dists = dists[np.triu_indices_from(dists, k=1)]
        sweep_spacing = max(np.percentile(dists, 25), 2.0)
    else:
        sweep_spacing = 3.0

    # Initialize robot pose outside the exploration area
    initial_pose = np.array([box_limits[0] - 5, box_limits[2] - 5, 0.0])
    true_pose = initial_pose.copy()
    ekf_slam.hypotheses[0].mu = initial_pose.copy()

    # Exploration control variables
    center_x = (box_limits[0] + box_limits[1]) / 2
    center_y = (box_limits[2] + box_limits[3]) / 2
    target_y = box_limits[2] + 1
    sweep_direction = 1
    in_exploration_area = False

    # Data logging
    estimated_path = []
    true_path = []
    landmark_history = []

    print("Starting EKF-SLAM simulation...")
    print(f"Exploration area: [{box_limits[0]:.1f}, {box_limits[1]:.1f}] x [{box_limits[2]:.1f}, {box_limits[3]:.1f}]")
    print(f"Number of landmarks: {len(landmarks)}")

    # Main simulation loop
    for t in range(300):
        x, y, theta = true_pose

        # Navigation logic
        if not in_exploration_area:
            # Navigate to exploration area
            if box_limits[0] <= x <= box_limits[1] and box_limits[2] <= y <= box_limits[3]:
                in_exploration_area = True
                print(f"Entered exploration area at step {t}")
            else:
                # Head towards center of exploration area
                dx = center_x - x
                dy = center_y - y
                desired_angle = np.arctan2(dy, dx)
                angle_diff = wrap_angle(desired_angle - theta)
                v = 1.0
                w = np.clip(angle_diff, -np.deg2rad(30), np.deg2rad(30))
                u = [v, w]
        else:
            # Exploration with obstacle avoidance
            repulsion = np.array([0.0, 0.0])
            for (lx, ly) in landmarks:
                dx_lm = lx - x
                dy_lm = ly - y
                dist = np.hypot(dx_lm, dy_lm)
                if dist < 2.0:  # Avoid getting too close to landmarks
                    repulsion -= np.array([dx_lm, dy_lm]) / (dist + 1e-3)

            # Sweeping pattern
            target_x = box_limits[1] - 1 if sweep_direction == 1 else box_limits[0] + 1
            dx = target_x - x
            dy = target_y - y

            # Switch direction when reaching edge
            if abs(dx) < 0.5:
                target_y += sweep_spacing
                if target_y > box_limits[3] - 1:
                    target_y = box_limits[2] + 1
                sweep_direction *= -1

            # Combine goal-seeking and obstacle avoidance
            goal_vec = np.array([dx, dy])
            heading_vec = goal_vec + 1.5 * repulsion
            
            desired_angle = np.arctan2(heading_vec[1], heading_vec[0])
            angle_diff = wrap_angle(desired_angle - theta)
            v = 1.0
            w = np.clip(angle_diff, -np.deg2rad(30), np.deg2rad(30))
            u = [v, w]

        # Simulate robot motion (ground truth)
        dt = ekf_slam.dt
        v, w = u
        dtheta = w * dt
        
        if np.abs(w) > 1e-6:
            # Circular arc motion
            dx = -v / w * np.sin(theta) + v / w * np.sin(theta + dtheta)
            dy = v / w * np.cos(theta) - v / w * np.cos(theta + dtheta)
        else:
            # Straight line motion
            dx = v * dt * np.cos(theta)
            dy = v * dt * np.sin(theta)
        
        true_pose[0] += dx
        true_pose[1] += dy
        true_pose[2] = wrap_angle(true_pose[2] + dtheta)

        # Simulate sensor observations
        observations = []
        FOV = np.deg2rad(60)  # 60-degree field of view
        
        for (lx, ly) in landmarks:
            dx = lx - true_pose[0]
            dy = ly - true_pose[1]
            r = np.sqrt(dx**2 + dy**2)
            bearing = wrap_angle(np.arctan2(dy, dx) - true_pose[2])
            
            # Check if landmark is within sensor FOV
            if -FOV / 2 <= bearing <= FOV / 2:
                # Add sensor noise
                noisy_r = r + np.random.normal(0, np.sqrt(ekf_slam.Q[0, 0]))
                noisy_phi = wrap_angle(bearing + np.random.normal(0, np.sqrt(ekf_slam.Q[1, 1])))
                observations.append(np.array([noisy_r, noisy_phi]))

        # EKF-SLAM update
        ekf_slam.update(u, observations)

        # Log data
        est_pose, est_landmarks = ekf_slam.get_state()
        estimated_path.append(est_pose[:2])
        true_path.append(true_pose[:2].copy())
        
        if est_landmarks is not None:
            landmark_history.append((est_landmarks.copy(), ekf_slam.get_covariance().copy()))
        
        # Real-time visualization
        if t % 10 == 0:  # Update plot every 10 steps
            ekf_slam.plot_landmarks()

    # Final results
    estimated_path = np.array(estimated_path)
    true_path = np.array(true_path)
    
    print(f"\nSimulation completed after {len(estimated_path)} steps")
    print(f"Final number of estimated landmarks: {len(landmark_history[-1][0]) if landmark_history else 0}")

    # Create final summary plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Plot paths
    plt.plot(estimated_path[:, 0], estimated_path[:, 1], 'b-', linewidth=2, 
             label='Estimated Path', alpha=0.8)
    plt.plot(true_path[:, 0], true_path[:, 1], 'k--', linewidth=2, 
             label='True Path', alpha=0.8)
    
    # Plot landmarks
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='g', marker='^', s=100, 
                label='True Landmarks', edgecolors='black', linewidth=1)

    # Plot final estimated landmarks with uncertainty
    if landmark_history:
        final_landmarks, final_cov = landmark_history[-1]
        plt.scatter(final_landmarks[:, 0], final_landmarks[:, 1], c='r', marker='x', 
                   s=100, linewidth=3, label='Estimated Landmarks')
        
        # Add uncertainty ellipses
        for i in range(final_landmarks.shape[0]):
            lm_idx = 3 + 2 * i
            cov = final_cov[lm_idx:lm_idx+2, lm_idx:lm_idx+2]
            mean = final_landmarks[i]
            
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 2 * np.sqrt(9.21 * eigenvals)  # 99% confidence
            
            ellipse = patches.Ellipse(xy=mean, width=width, height=height, angle=angle,
                                    edgecolor='r', facecolor='none', linewidth=1, alpha=0.7)
            ax.add_patch(ellipse)

    # Mark start and end positions
    plt.plot(true_path[0, 0], true_path[0, 1], 'go', markersize=10, label='Start')
    plt.plot(true_path[-1, 0], true_path[-1, 1], 'ro', markersize=10, label='End')

    plt.title("EKF-SLAM Results: Autonomous Exploration with Multiple Hypothesis Tracking", fontsize=14)
    plt.xlabel("X (m)", fontsize=12)
    plt.ylabel("Y (m)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the simulation
    run_ekf_slam_simulation()