import numpy as np

class EKF_SLAM:
    """
    Extended Kalman Filter SLAM implementation with unknown correspondences.
    This class provides functionality to track robot pose and landmark positions
    simultaneously while building a map of the environment.
    """
    
    def __init__(self):
        """
        Initialize the EKF SLAM algorithm.
        - mu: State vector [x, y, theta, landmark1_x, landmark1_y, ...]
        - Sigma: Covariance matrix
        - R: Motion noise covariance
        - Q: Measurement noise covariance
        """
        # Initialize state vector with robot pose [x, y, theta]
        self.mu = np.zeros(3)
        # Initialize covariance matrix for initial state
        self.Sigma = np.zeros((3, 3))
        # Default motion noise (can be modified by setter method)
        self.R = np.diag([0.1, 0.1, np.deg2rad(5)]) ** 2
        # Default measurement noise (can be modified by setter method)
        self.Q = np.diag([0.5, np.deg2rad(2.0)]) ** 2
        # Time increment
        self.dt = 1.0
        # Threshold for adding new landmarks (chi-square value with p=0.99)
        self.alpha_threshold = 9.21
        # Minimum distance between landmarks to avoid duplicates
        self.min_landmark_distance = 0.5
        
    def set_noise_parameters(self, R=None, Q=None):
        """
        Set the noise covariance matrices.
        
        Args:
            R: 3x3 motion noise covariance matrix
            Q: 2x2 measurement noise covariance matrix
        """
        if R is not None:
            self.R = R
        if Q is not None:
            self.Q = Q
            
    def set_time_step(self, dt):
        """Set the time step for prediction."""
        self.dt = dt
        
    def set_landmark_threshold(self, alpha_threshold, min_distance=None):
        """
        Set the threshold for adding new landmarks.
        
        Args:
            alpha_threshold: Mahalanobis distance threshold (chi-square value)
            min_distance: Minimum Euclidean distance between landmarks
        """
        self.alpha_threshold = alpha_threshold
        if min_distance is not None:
            self.min_landmark_distance = min_distance
            
    def get_state(self):
        """
        Get the current state estimate.
        
        Returns:
            tuple: (robot_pose, landmarks)
                - robot_pose: numpy array [x, y, theta]
                - landmarks: numpy array of shape (n_landmarks, 2)
        """
        robot_pose = self.mu[:3].copy()
        landmarks = None
        
        if len(self.mu) > 3:
            landmarks = self.mu[3:].reshape(-1, 2)
            
        return robot_pose, landmarks
    
    def get_covariance(self):
        """Get the current state covariance matrix."""
        return self.Sigma.copy()
        
    @staticmethod
    def _wrap_angle(angle):
        """
        Normalize angle to [-pi, pi).
        
        Args:
            angle: Angle in radians
            
        Returns:
            Normalized angle
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
            
    def update(self, u, z):
        """
        Perform one EKF SLAM update step.
        
        Args:
            u: Control input [v, w] - linear and angular velocity
            z: List of measurements, each [r, phi] - range and bearing
            
        Returns:
            tuple: (robot_pose, landmarks)
        """
        self.mu, self.Sigma = self._ekf_slam_step(self.mu, self.Sigma, u, z, 
                                                 self.R, self.Q, self.dt, 
                                                 self.alpha_threshold)
        return self.get_state()
        
    def _ekf_slam_step(self, mu, Sigma, u, z, R, Q, dt, alpha_threshold):
        """
        Main EKF SLAM algorithm with unknown correspondences.
        
        Args:
            mu: State vector [x, y, theta, landmark1_x, landmark1_y, ...]
            Sigma: Covariance matrix
            u: Control input [v, w] - linear and angular velocity
            z: List of measurements, each [r, phi] - range and bearing
            R: Motion noise covariance
            Q: Measurement noise covariance
            dt: Time step
            alpha_threshold: Threshold for adding new landmarks
            
        Returns:
            tuple: (new_mu, new_Sigma)
        """
        # Get number of landmarks in the state
        n_landmarks = (len(mu) - 3) // 2
        
        # Create matrix to extract robot pose from state
        Fx = np.hstack([np.eye(3), np.zeros((3, 2 * n_landmarks))])

        # Extract control inputs
        v, w = u
        theta = mu[2]
        dtheta = w * dt
        
        # Motion model: calculate position change based on velocity and steering
        if abs(w) > 1e-6:
            # Circular motion model when angular velocity is significant
            dx = -v / w * np.sin(theta) + v / w * np.sin(theta + dtheta)
            dy = v / w * np.cos(theta) - v / w * np.cos(theta + dtheta)
        else:
            # Straight-line model when angular velocity is negligible
            dx = v * dt * np.cos(theta)
            dy = v * dt * np.sin(theta)

        # Prediction step (motion update)
        mu_bar = mu.copy()
        mu_bar[0] += dx  # Update x
        mu_bar[1] += dy  # Update y
        mu_bar[2] = self._wrap_angle(mu_bar[2] + dtheta)  # Update and normalize theta

        # Jacobian of motion model with respect to state
        if abs(w) > 1e-6:
            Gx = np.array([
                [0, 0, v/w * np.cos(theta) - v/w * np.cos(theta + dtheta)],
                [0, 0, v/w * np.sin(theta) - v/w * np.sin(theta + dtheta)],
                [0, 0, 0]
            ])
        else:
            Gx = np.array([
                [0, 0, -v * dt * np.sin(theta)],
                [0, 0, v * dt * np.cos(theta)],
                [0, 0, 0]
            ])

        # Complete Jacobian of motion model
        G = np.eye(len(mu)) + Fx.T @ Gx @ Fx
        
        # Prediction covariance update
        Sigma_bar = G @ Sigma @ G.T + Fx.T @ R @ Fx

        # Process each measurement for update or landmark addition
        for i in range(len(z)):
            r_i, phi_i = z[i]  # Range and bearing measurement
            z_i = np.array([r_i, phi_i])

            # Variables to keep track of best landmark match
            min_mahalanobis = float('inf')
            best_dz = None
            best_H = None
            best_K = None

            # Check all existing landmarks for correspondence
            Nt = (len(mu_bar) - 3) // 2
            for j in range(Nt):
                # Index of current landmark in state vector
                lm_index = 3 + 2 * j
                
                # Calculate difference between landmark and robot position
                delta = mu_bar[lm_index:lm_index+2] - mu_bar[0:2]
                q = delta.T @ delta  # Squared distance
                sqrt_q = np.sqrt(q)  # Distance
                
                # Expected measurement if this landmark is observed
                z_hat = np.array([
                    sqrt_q,  # Expected range
                    self._wrap_angle(np.arctan2(delta[1], delta[0]) - mu_bar[2])  # Expected bearing
                ])
                
                # Measurement innovation (difference between actual and expected)
                dz = z_i - z_hat
                dz[1] = self._wrap_angle(dz[1])  # Normalize angle difference

                # Create matrix to extract relevant state variables
                Fxj = np.zeros((5, len(mu_bar)))
                Fxj[0:3, 0:3] = np.eye(3)  # Robot pose
                Fxj[3, lm_index] = 1      # Landmark x
                Fxj[4, lm_index+1] = 1    # Landmark y

                # Jacobian of measurement model
                H = (1 / q) * np.array([
                    [-sqrt_q * delta[0], -sqrt_q * delta[1], 0, sqrt_q * delta[0], sqrt_q * delta[1]],
                    [delta[1], -delta[0], -q, -delta[1], delta[0]]
                ]) @ Fxj

                # Innovation covariance
                S = H @ Sigma_bar @ H.T + Q
                
                # Calculate Mahalanobis distance to check correspondence
                mahalanobis = dz.T @ np.linalg.inv(S) @ dz

                # Keep track of best match
                if mahalanobis < min_mahalanobis:
                    min_mahalanobis = mahalanobis
                    best_dz = dz
                    best_H = H
                    best_K = Sigma_bar @ H.T @ np.linalg.inv(S)

            # If no good match found, create new landmark
            if min_mahalanobis > alpha_threshold:
                # Convert polar measurement to global Cartesian coordinates
                lx = mu_bar[0] + r_i * np.cos(phi_i + mu_bar[2])
                ly = mu_bar[1] + r_i * np.sin(phi_i + mu_bar[2])

                # Check if the new landmark is too close to existing ones
                too_close = False
                for j in range((len(mu_bar) - 3) // 2):
                    idx = 3 + 2*j
                    existing_lx = mu_bar[idx]
                    existing_ly = mu_bar[idx+1]
                    dist = np.hypot(existing_lx - lx, existing_ly - ly)
                    if dist < self.min_landmark_distance:
                        too_close = True
                        break

                # Add new landmark if not too close to existing ones
                if not too_close:
                    # Extend state vector with new landmark
                    mu_bar = np.append(mu_bar, [lx, ly])
                    
                    # Extend covariance matrix
                    new_Sigma = np.zeros((len(mu_bar), len(mu_bar)))
                    new_Sigma[:Sigma_bar.shape[0], :Sigma_bar.shape[1]] = Sigma_bar
                    new_Sigma[-2:, -2:] = np.diag([1, 1])  # Initial landmark uncertainty
                    Sigma_bar = new_Sigma

            # Update state with matched landmark
            else:
                # Kalman update equations
                mu_bar = mu_bar + best_K @ best_dz
                mu_bar[2] = self._wrap_angle(mu_bar[2])  # Normalize robot heading
                Sigma_bar = (np.eye(len(mu_bar)) - best_K @ best_H) @ Sigma_bar

        return mu_bar, Sigma_bar


# Example usage (not executed when imported)
if __name__ == "__main__":
    # Initialize SLAM object
    slam = EKF_SLAM()
    
    # Set parameters if needed
    slam.set_time_step(0.1)  # 100ms time step
    
    # Example control and measurement
    u = [1.0, np.deg2rad(5)]  # Move forward at 1m/s with 5deg/s rotation
    z = [[5.0, np.deg2rad(30)], [7.0, np.deg2rad(-45)]]  # Two landmark measurements
    
    # Update SLAM state
    robot_pose, landmarks = slam.update(u, z)
    
    # Get results
    print(f"Robot pose: x={robot_pose[0]:.2f}, y={robot_pose[1]:.2f}, θ={np.rad2deg(robot_pose[2]):.2f}°")
    if landmarks is not None:
        print(f"Detected {len(landmarks)} landmarks:")
        for i, lm in enumerate(landmarks):
            print(f"  Landmark {i+1}: ({lm[0]:.2f}, {lm[1]:.2f})")