import numpy as np
import matplotlib.pyplot as plt

class EKF_SLAM:
	"""
	Extended Kalman Filter SLAM implementation with unknown correspondences,
	plus automatic merging of landmarks that end up too close to each other
	(to avoid phantom / duplicate landmarks).
	"""
	
	def __init__(self):
		"""
		Initialize the EKF SLAM algorithm.
		- mu: State vector [x, y, theta, lm1_x, lm1_y, ...]
		- Sigma: Covariance matrix
		"""
		self.mu = np.zeros(3)                   # robot pose [x,y,θ]
		self.Sigma = np.eye(3)                  # initial covariance
		self.R = np.diag([0.1, 0.1, np.deg2rad(5)])**2
		self.Q = np.diag([1.0, np.deg2rad(5)])**2
		self.dt = 0.032
		self.thresholds = {0.1:4.61,0.05:5.99,0.01:9.21,0.001:13.82}
		self.alpha_threshold = self.thresholds[0.001]
		self.min_landmark_distance = 0.5        # min merge distance

	def set_noise_parameters(self, R=None, Q=None):
		if R is not None: self.R = R
		if Q is not None: self.Q = Q

	def set_time_step(self, dt):
		self.dt = dt

	def set_landmark_threshold(self, alpha_threshold, min_distance=None):
		self.alpha_threshold = alpha_threshold
		if min_distance is not None:
			self.min_landmark_distance = min_distance

	def get_state(self):
		pose = self.mu[:3].copy()
		lms  = self.mu[3:].reshape(-1,2).copy() if len(self.mu)>3 else None
		return pose, lms

	def get_covariance(self):
		return self.Sigma.copy()

	@staticmethod
	def _wrap_angle(a):
		return (a + np.pi)%(2*np.pi) - np.pi

	def update(self, u, z):
		"""
		1) EKF prediction + correction (or new‐landmark addition)
		2) Merge any landmarks that ended up within min_landmark_distance
		"""
		mu_bar, Sigma_bar = self._ekf_slam_step(
			self.mu, self.Sigma, u, z, self.R, self.Q, self.dt, self.alpha_threshold
		)

		# --- MERGE CLOSE LANDMARKS ---
		mu_merged, Sigma_merged = self._merge_close_landmarks(
			mu_bar, Sigma_bar, self.min_landmark_distance
		)

		self.mu, self.Sigma = mu_merged, Sigma_merged
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
	def _merge_close_landmarks(self, mu, Sigma, distance_threshold):
		"""
		Cluster any landmarks closer than distance_threshold
		and merge them by averaging their positions.
		Covariances of merged clusters are taken from the first landmark.
		"""
		if len(mu) <= 3:
			return mu, Sigma  # no landmarks to merge

		# extract and reshape landmarks
		lms = mu[3:].reshape(-1, 2)
		N  = lms.shape[0]
		used = np.zeros(N, dtype=bool)
		new_lms = []
		new_indices = []

		for i in range(N):
			if used[i]:
				continue
			# start a new cluster with landmark i
			cluster = [lms[i]]
			used[i] = True
			for j in range(i+1, N):
				if not used[j] and np.linalg.norm(lms[i] - lms[j]) < distance_threshold:
					cluster.append(lms[j])
					used[j] = True

			# average cluster
			merged = np.mean(cluster, axis=0)
			new_lms.append(merged)
			new_indices.append(i)

		# rebuild state vector
		new_mu = np.hstack([mu[:3], np.array(new_lms).ravel()])

		# rebuild covariance
		dim = len(new_mu)
		new_Sigma = np.zeros((dim, dim))
		# copy robot‐pose covariance
		new_Sigma[:3, :3] = Sigma[:3, :3]

		# for each kept cluster, copy its old 2×2 block
		for k, old_i in enumerate(new_indices):
			old_idx = 3 + 2*old_i
			new_idx = 3 + 2*k
			new_Sigma[new_idx:new_idx+2, new_idx:new_idx+2] = \
				Sigma[old_idx:old_idx+2, old_idx:old_idx+2]

		return new_mu, new_Sigma

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

	def plot_landmarks(self, landmarks, robot_pose):
		plt.cla()
		if landmarks is not None:
			plt.scatter(landmarks[:,0], landmarks[:,1], c='r', label='Landmarks')
		plt.quiver(robot_pose[0], robot_pose[1],
				   np.cos(robot_pose[2]), np.sin(robot_pose[2]),
				   angles='xy', scale_units='xy', scale=1,
				   color='b', label='Robot')
		plt.xlim(-10,10); plt.ylim(-10,10)
		plt.xlabel('X'); plt.ylabel('Y')
		plt.title('EKF SLAM Map')
		plt.grid(); plt.pause(0.001)
