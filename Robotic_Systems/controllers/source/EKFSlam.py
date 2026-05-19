import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') # Use Qt5Agg for interactive window in Docker
import matplotlib.pyplot as plt

class EKF_SLAM:
	def __init__(self):
		self.hypotheses = [self.Hypothesis(np.zeros(3), np.eye(3))]
		self.R = np.diag([0.005, 0.005, np.deg2rad(0.2)]) ** 2
		self.Q = np.diag([0.1, np.deg2rad(5)]) ** 2
		self.dt = 0.032
		self.alpha_threshold = 9.21
		self.min_landmark_distance = 0.6
		self.max_hypotheses = 5
		
	class Hypothesis:
		def __init__(self, mu, Sigma, score=0.0):
			self.mu = mu.copy()
			self.Sigma = Sigma.copy()
			self.score = score

		def clone(self):
			return EKF_SLAM.Hypothesis(self.mu.copy(), self.Sigma.copy(), self.score)

	@staticmethod
	def wrap_angle(angle):
		return (angle + np.pi) % (2 * np.pi) - np.pi

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
		best = self.hypotheses[0]
		pose = best.mu[:3].copy()
		lms = best.mu[3:].reshape(-1, 2).copy() if len(best.mu) > 3 else None
		return pose, lms

	def get_covariance(self):
		return self.hypotheses[0].Sigma.copy()

	def predict(self, u):
		new_hypotheses = []
		for hyp in self.hypotheses:
			mu_bar, Sigma_bar = self._motion_update(hyp.mu, hyp.Sigma, u)
			new_hypotheses.append(self.Hypothesis(mu_bar, Sigma_bar, hyp.score))
		self.hypotheses = new_hypotheses

	def update(self, u, z):
		self.predict(u)
		if z:
			self.correct(z)

	def correct(self, z):
		new_hypotheses = []
		# CORRECT — observations applied sequentially
		for hyp in self.hypotheses:
			current_mu, current_Sigma = hyp.mu.copy(), hyp.Sigma.copy()
			current_score = hyp.score
			for zi in z:
				associations = self._associate(current_mu, current_Sigma, zi)
				if not associations:
					current_mu, current_Sigma = self._add_new_landmark(current_mu, current_Sigma, zi)
				else:
					best = min(associations, key=lambda a: a[4])  # pick lowest Mahalanobis
					j, dz, H, K, maha = best
					current_mu = current_mu + K @ dz
					current_mu[2] = self.wrap_angle(current_mu[2])
					current_Sigma = (np.eye(len(current_mu)) - K @ H) @ current_Sigma
					current_score += (self.alpha_threshold - maha)
			new_hypotheses.append(self.Hypothesis(current_mu, current_Sigma, current_score))
		self.hypotheses = sorted(new_hypotheses, key=lambda h: h.score, reverse=True)[:self.max_hypotheses]
		best = self.hypotheses[0]
		best.mu, best.Sigma = self._merge_close_landmarks(best.mu, best.Sigma, self.min_landmark_distance)

	def _motion_update(self, mu, Sigma, u):
		v, w = u
		theta = mu[2]
		dtheta = w * self.dt

		if np.isclose(w, 0.0):
			dx = v * self.dt * np.cos(theta)
			dy = v * self.dt * np.sin(theta)
		else:
			dx = -v / w * np.sin(theta) + v / w * np.sin(theta + dtheta)
			dy = v / w * np.cos(theta) - v / w * np.cos(theta + dtheta)

		mu_bar = mu.copy()
		mu_bar[0] += dx
		mu_bar[1] += dy
		mu_bar[2] = self.wrap_angle(mu_bar[2] + dtheta)

		n_landmarks = (len(mu) - 3) // 2
		Fx = np.hstack([np.eye(3), np.zeros((3, 2 * n_landmarks))])

		if np.isclose(w, 0.0):
			Gx = np.array([
				[0, 0, -v * self.dt * np.sin(theta)],
				[0, 0, v * self.dt * np.cos(theta)],
				[0, 0, 0]
			])
		else:
			Gx = np.array([
				[0, 0, v / w * np.cos(theta) - v / w * np.cos(theta + dtheta)],
				[0, 0, v / w * np.sin(theta) - v / w * np.sin(theta + dtheta)],
				[0, 0, 0]
			])

		G = np.eye(len(mu)) + Fx.T @ Gx @ Fx
		Sigma_bar = G @ Sigma @ G.T + Fx.T @ self.R @ Fx
		return mu_bar, Sigma_bar

	def _associate(self, mu, Sigma, z):
		associations = []
		Nt = (len(mu) - 3) // 2
		for j in range(Nt):
			idx = 3 + 2 * j
			delta = mu[idx:idx + 2] - mu[0:2]
			q = delta.T @ delta
			sqrt_q = np.sqrt(q)
			z_hat = np.array([sqrt_q, self.wrap_angle(np.arctan2(delta[1], delta[0]) - mu[2])])
			dz = z - z_hat
			dz[1] = self.wrap_angle(dz[1])

			Fxj = np.zeros((5, len(mu)))
			Fxj[0:3, 0:3] = np.eye(3)
			Fxj[3, idx] = 1
			Fxj[4, idx + 1] = 1

			H = (1 / q) * np.array([
				[-sqrt_q * delta[0], -sqrt_q * delta[1], 0, sqrt_q * delta[0], sqrt_q * delta[1]],
				[delta[1], -delta[0], -q, -delta[1], delta[0]]
			]) @ Fxj

			S = H @ Sigma @ H.T + self.Q
			maha = dz.T @ np.linalg.inv(S) @ dz

			if maha < self.alpha_threshold:
				K = Sigma @ H.T @ np.linalg.inv(S)
				associations.append((j, dz, H, K, maha))

		if associations:
			return [min(associations, key=lambda a: a[4])]
		return []


	def _add_new_landmark(self, mu, Sigma, z):
		r, phi = z
		lx = mu[0] + r * np.cos(phi + mu[2])
		ly = mu[1] + r * np.sin(phi + mu[2])
		mu_new = np.append(mu, [lx, ly])
		Sigma_new = np.zeros((len(mu_new), len(mu_new)))
		Sigma_new[:len(Sigma), :len(Sigma)] = Sigma
		Sigma_new[-2:, -2:] = np.diag([0.5, 0.5])
		return mu_new, Sigma_new


	def _merge_close_landmarks(self, mu, Sigma, distance_threshold):
		if len(mu) <= 3:
			return mu, Sigma

		lms = mu[3:].reshape(-1, 2)
		N = lms.shape[0]
		used = np.zeros(N, dtype=bool)
		new_lms = []
		keep_indices = []

		print(f"[Merge] Checking {N} landmarks with threshold {distance_threshold:.2f} m")

		for i in range(N):
			if used[i]:
				continue

			cluster = [lms[i]]
			cluster_indices = [i]
			used[i] = True

			for j in range(i + 1, N):
				if used[j]:
					continue

				diff = lms[i] - lms[j]
				euclidean_dist = np.linalg.norm(diff)

				if euclidean_dist < distance_threshold:
					cluster.append(lms[j])
					cluster_indices.append(j)
					used[j] = True

			merged = np.mean(cluster, axis=0)
			new_lms.append(merged)
			keep_indices.append(i)

			if len(cluster) > 1:
				print(f"[Merge] Clustered landmarks {cluster_indices} at {np.round(cluster, 2)}")
				print(f"        → Merged at {np.round(merged, 2)}")

		# Rebuild state vector and covariance
		new_mu = mu[:3]
		for lm in new_lms:
			new_mu = np.append(new_mu, lm)

		new_Sigma = np.zeros((len(new_mu), len(new_mu)))
		new_Sigma[:3, :3] = Sigma[:3, :3]

		for i, old_i in enumerate(keep_indices):
			oi = 3 + 2 * old_i
			ni = 3 + 2 * i
			# pose <-> landmark
			new_Sigma[:3, ni:ni+2] = Sigma[:3, oi:oi+2]
			new_Sigma[ni:ni+2, :3] = Sigma[oi:oi+2, :3]
			# landmark <-> landmark
			for j, old_j in enumerate(keep_indices):
				oj = 3 + 2 * old_j
				nj = 3 + 2 * j
				new_Sigma[ni:ni+2, nj:nj+2] = Sigma[oi:oi+2, oj:oj+2]

		print(f"[Merge] {len(new_lms)} landmarks after merging\n{'-'*40}")
		return new_mu, new_Sigma
	
	def plot_landmarks(self):
		pose, landmarks = self.get_state()
		plt.cla()
		if landmarks is not None:
			plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', label='Landmarks')
		plt.quiver(pose[0], pose[1],
				   np.cos(pose[2]), np.sin(pose[2]),
				   angles='xy', scale_units='xy', scale=0.5,
				   color='b', label='Robot')
		plt.xlim(-5, 5)
		plt.ylim(-5, 5)
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.title('EKF SLAM Map')
		plt.grid()
		plt.legend()
		plt.pause(0.001)

