import numpy as np
import matplotlib.pyplot as plt

class EKF_SLAM:
	"""
	Extended Kalman Filter SLAM for a differential‐drive robot, using wheel encoders.
	- mu:   State vector [x, y, θ, l1_x, l1_y, l2_x, l2_y, …]
	- Sigma: Covariance matrix
	- R:    Motion noise covariance (in wheel units)
	- Q:    Measurement noise covariance
	"""

	def __init__(
		self,
		wheel_radius=0.042,
		wheel_base=0.194,
		R=None,
		Q=None,
	):
		"""
		Args:
			wheel_radius (float): radius of each wheel [m]. If provided,
								   controls u = [ω_left, ω_right] (rad/s).
			wheel_base   (float): distance between wheels [m].
			R (3×3 array):     motion noise covariance in (v,w)
			Q (2×2 array):     measurement noise covariance
		"""
		# -- core SLAM state --
		self.mu    = np.zeros(3)               # [x, y, θ]
		self.Sigma = np.eye(3)                 # large initial uncertainty

		# -- motion & measurement noise --
		self.R = np.diag([0.1, 0.1, np.deg2rad(0.1)])**2 if R is None else R
		self.Q = np.diag([1.0, np.deg2rad(5.0)])**2 if Q is None else Q

		# -- timing & thresholds --
		self.dt = 0.032
		self.thresholds = {0.1:4.61, 0.05:5.99, 0.01:9.21, 0.001:13.82}
		self.alpha_threshold = self.thresholds[0.001]
		self.min_landmark_distance = 0.6

		# -- differential drive parameters --
		self.wheel_radius = wheel_radius
		self.wheel_base   = wheel_base
		# if both are given, we'll convert encoder readings → (v,w)
		self.use_wheel_model = (wheel_radius is not None and wheel_base is not None)

	def set_wheel_parameters(self, wheel_radius, wheel_base):
		"""
		Provide your robot’s wheel geometry so that
		update() will take encoder readings [ω_l, ω_r].

		Args:
			wheel_radius (float): radius of each wheel [m]
			wheel_base (float):   distance between wheels [m]
		"""
		self.wheel_radius   = wheel_radius
		self.wheel_base     = wheel_base
		self.use_wheel_model = True

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
		One full EKF‐SLAM cycle: predict + correct / add‐landmarks.

		Args:
		  u: if wheel params set → [ω_left, ω_right] in rad/s
			 else             → [v, w] in [m/s, rad/s]
		  z: list of measurements [ [r, φ], … ]

		Returns:
		  robot_pose, landmarks
		"""
		# -- convert wheel encoder readings → v, w if needed --
		if self.use_wheel_model:
			omega_l, omega_r = u
			v = self.wheel_radius * (omega_r + omega_l) / 2.0
			w = self.wheel_radius * (omega_r - omega_l) / self.wheel_base
			u_vw = [v, w]
		else:
			u_vw = u

		# execute the EKF‐SLAM step
		self.mu, self.Sigma = self._ekf_slam_step(
			self.mu, self.Sigma, u_vw, z, self.R, self.Q, self.dt, self.alpha_threshold
		)
		return self.get_state()
	
	@staticmethod
	def _kl_divergence(mean_p, cov_p, mean_q, cov_q):
		"""
		Symmetric KL divergence between two Gaussians P ~ N(mean_p, cov_p) and
		Q ~ N(mean_q, cov_q):
		
		   D_sym(P‖Q) = 0.5 [ D(P‖Q) + D(Q‖P) ]
		"""
		k = mean_p.shape[0]
		
		# Calculate D(P‖Q)
		inv_q = np.linalg.inv(cov_q)
		delta_pq = mean_q - mean_p
		term1_pq = np.trace(inv_q @ cov_p)
		term2_pq = delta_pq.T @ inv_q @ delta_pq
		term3_pq = np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p))
		kl_pq = 0.5 * (term1_pq + term2_pq - k + term3_pq)
		
		# Calculate D(Q‖P)
		inv_p = np.linalg.inv(cov_p)
		delta_qp = mean_p - mean_q
		term1_qp = np.trace(inv_p @ cov_q)
		term2_qp = delta_qp.T @ inv_p @ delta_qp
		term3_qp = np.log(np.linalg.det(cov_p) / np.linalg.det(cov_q))
		kl_qp = 0.5 * (term1_qp + term2_qp - k + term3_qp)
		
		return 0.5 * (kl_pq + kl_qp)


	def _ekf_slam_step(self, mu, Sigma, u, z, R, Q, dt, alpha_threshold ):
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
				[0, 0, -v/w * np.cos(theta) + v/w * np.cos(theta + dtheta)],
				[0, 0, -v/w * np.sin(theta) + v/w * np.sin(theta + dtheta)],
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
		for (r_i, phi_i) in z:
			z_i = np.array([r_i, phi_i])
			best_kl = float('inf')
			best_dz = None
			best_H = None
			best_K = None

			Nt = (len(mu_bar) - 3) // 2
			for j in range(Nt):
				idx = 3 + 2*j
				delta = mu_bar[idx:idx+2] - mu_bar[0:2]
				q = delta.T @ delta
				sqrt_q = np.sqrt(q)

				# expected measurement
				z_hat = np.array([
					sqrt_q,
					self._wrap_angle(np.arctan2(delta[1], delta[0]) - mu_bar[2])
				])

				# Innovation Jacobian H (same as before)
				Fxj = np.zeros((5, len(mu_bar)))
				Fxj[0:3, 0:3] = np.eye(3)
				Fxj[3, idx]   = 1
				Fxj[4, idx+1] = 1

				H = (1 / q) * np.array([
					[-sqrt_q * delta[0], -sqrt_q * delta[1], 0, sqrt_q * delta[0], sqrt_q * delta[1]],
					[ delta[1],         -delta[0],        -q, -delta[1],         delta[0]]
				]) @ Fxj

				S = H @ Sigma_bar @ H.T + Q

				# compute symmetric KL divergence between:
				#   P = N(z_hat, S)   and   Q = N(z_i, Q)
				kl_div = self._kl_divergence(z_hat, S, z_i, Q)

				if kl_div < best_kl:
					best_kl  = kl_div
					best_dz  = z_i - z_hat
					best_dz[1] = self._wrap_angle(best_dz[1])
					best_H   = H
					best_K   = Sigma_bar @ H.T @ np.linalg.inv(S)

			# decide: new landmark if KL too large
			if best_kl > alpha_threshold:
				lx = mu_bar[0] + r_i * np.cos(phi_i + mu_bar[2])
				ly = mu_bar[1] + r_i * np.sin(phi_i + mu_bar[2])

				# check proximity to existing
				too_close = False
				for j in range((len(mu_bar)-3)//2):
					ex, ey = mu_bar[3+2*j:3+2*j+2]
					if np.hypot(ex-lx, ey-ly) < self.min_landmark_distance:
						too_close = True
						break

				if not too_close:
					# append landmark
					mu_bar = np.hstack([mu_bar, [lx, ly]])
					new_S = np.zeros((len(mu_bar), len(mu_bar)))
					old_n = Sigma_bar.shape[0]
					new_S[:old_n, :old_n] = Sigma_bar
					new_S[-2:, -2:] = np.eye(2)  # initial high uncertainty
					Sigma_bar = new_S

			else:
				# standard EKF update
				mu_bar    = mu_bar + best_K @ best_dz
				mu_bar[2] = self._wrap_angle(mu_bar[2])
				Sigma_bar = (np.eye(len(mu_bar)) - best_K @ best_H) @ Sigma_bar

		return mu_bar, Sigma_bar
	
	
	def plot_landmarks(self, landmarks,robot_pose):
		"""
		Plot the landmarks and robot pose.

		Args:
			landmarks: List of landmark positions (r,phi)
			robot_pose: Robot pose [x, y, theta]
		"""
		
		plt.cla()
		plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', label='Landmarks')
		plt.quiver(robot_pose[0], robot_pose[1], np.cos(robot_pose[2]), np.sin(robot_pose[2]), 
				   angles='xy', scale_units='xy', scale=1, color='b', label='Robot Pose')
		plt.xlim(-10, 10)
		plt.ylim(-10, 10)
		plt.xlabel('X Position')
		plt.ylabel('Y Position')
		plt.title('EKF SLAM Landmarks and Robot Pose')
		# plt.legend()
		plt.grid()
		# wait for one second and clear the plot
		plt.pause(0.001)
		# Uncomment the following line to show the plot
		
	def plot_debug_landmarks(self, landmarks, robot_pose):
		"""
		Plot the landmarks and robot pose.

		Args:
			landmarks: List of landmark positions (r,phi)
			robot_pose: Robot pose [x, y, theta]
		"""
		
		plt.cla()
		plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', label='Landmarks')
		plt.quiver(robot_pose[0], robot_pose[1], np.cos(robot_pose[2]), np.sin(robot_pose[2]), 
				   angles='xy', scale_units='xy', scale=1, color='b', label='Robot Pose')
		plt.xlim(-10, 10)
		plt.ylim(-10, 10)
		plt.xlabel('X Position')
		plt.ylabel('Y Position')
		plt.title('EKF SLAM Landmarks and Robot Pose')
		# plt.legend()
		plt.grid()

		
# Example usage (not executed when imported)
# if __name__ == "__main__":
#     # Initialize SLAM object
#     slam = EKF_SLAM()
	
#     # Set parameters if needed
#     slam.set_time_step(0.1)  # 100ms time step
	
#     # Example control and measurement
#     u = [0, np.deg2rad(10)]  # Move forward at 1m/s with 5deg/s rotation
#     z = [
#         [5.0, np.deg2rad(0)], 
#         # [7.0, np.deg2rad(-45)], 
#         # [3.0, np.deg2rad(60)], 
#         [2.0, np.deg2rad(-180)]
#          ]
	
#     # Update SLAM state
#     robot_pose, landmarks = slam.update(u, z)
	
#     # Get results
#     print(f"Robot pose: x={robot_pose[0]:.2f}, y={robot_pose[1]:.2f}, θ={np.rad2deg(robot_pose[2]):.2f}°")
#     if landmarks is not None:
#         print(f"Detected {len(landmarks)} landmarks:")
#         for i, lm in enumerate(landmarks):
#             print(f"  Landmark {i+1}: ({lm[0]:.2f}, {lm[1]:.2f})")

	
#     u = [0, np.deg2rad(10)]  # Move forward at 1m/s with 5deg/s rotation
	
#     # Update SLAM state
#     robot_pose, landmarks = slam.update(u, z)
	
#     # Get results
#     print(f"Robot pose: x={robot_pose[0]:.2f}, y={robot_pose[1]:.2f}, θ={np.rad2deg(robot_pose[2]):.2f}°")
#     if landmarks is not None:
#         print(f"Detected {len(landmarks)} landmarks:")
#         for i, lm in enumerate(landmarks):
#             print(f"  Landmark {i+1}: ({lm[0]:.2f}, {lm[1]:.2f})")
#     # Plot landmarks and robot pose
#     slam.plot_landmarks(landmarks, robot_pose)
#     plt.show()
