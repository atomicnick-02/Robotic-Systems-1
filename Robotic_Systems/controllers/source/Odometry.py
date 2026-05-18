from controller import Robot
import numpy as np

def angle_normalize(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))



class Odometry:
	def __init__(self, robot, time_step_ms):
		"""
		Initialize Odometry with Webots Robot instance.
		Pose is [x, y, theta], and velocity [v, w].
		"""
		# Robot parameters
		self.wheel_radius   = 0.043     # m
		self.wheel_distance = 0.220     # m between wheels
		
		# Time step (s)
		self.dt = time_step_ms / 1000.0
		
		# State
		self.prev_enc = None            # will hold [left, right] on first update
		self.position = np.zeros(3)     # [x, y, θ]
		self.velocity = np.zeros(2)     # [v (m/s), w (rad/s)]
		
		print("# Odometry initialized successfully #")

	def update_from_encoders(self, left_encoder, right_encoder):
		"""
		Read new encoder readings and update pose + velocity.

		Returns:
			position: np.array([x, y, θ])
			velocity: np.array([v (m/s), ω (rad/s)])
		"""
		enc = np.array([left_encoder, right_encoder])
		print(f"dt = {self.dt:.4f} s")
		# First call: just initialize previous encoders
		if self.prev_enc is None:
			self.prev_enc = enc.copy()
			return self.position.copy(), self.velocity.copy()

		# 1) Compute change in encoder ticks
		delta_ticks = enc - self.prev_enc

		# 2) Convert to wheel rotation angles (rad)
		dphi_L = delta_ticks[0]
		dphi_R = delta_ticks[1]

		# 3) Compute linear distances traveled by each wheel (m)
		dL = dphi_L * self.wheel_radius
		dR = dphi_R * self.wheel_radius

		# 4) Center displacement and heading change
		d_center = (dR + dL) / 2.0            # m
		d_theta = (dR - dL) / self.wheel_distance  # rad

		# 5) Mid‐point integration for x, y
		theta_old = self.position[2]
		theta_mid = angle_normalize(theta_old + d_theta / 2.0)

		self.position[0] += d_center * np.cos(theta_mid)
		self.position[1] += d_center * np.sin(theta_mid)
		self.position[2] = angle_normalize(theta_old + d_theta)

		# 6) Compute velocities (divide distance by dt)
		v = d_center / self.dt
		w = d_theta / self.dt
		print(f"v: {v:.4f} m/s, w: {w:.4f} rad/s")
		self.velocity = np.array([v, w])

		# 7) Save for next iteration
		self.prev_enc = enc.copy()
		print(f"ΔL: {dL:.4f}, ΔR: {dR:.4f}, θ: {np.degrees(self.position[2]):.2f}°")
		return self.position.copy(), self.velocity.copy()

	def get_velocity(self):
		"""
		Returns the last computed velocities [v, w].
		"""
		return self.velocity.copy()

	def get_wheel_velocities(self):
		"""Returns (ω_L, ω_R) in rad/s from the last update."""
		# velocity already computed: v = (vR + vL)/2, w = (vR - vL)/L
		# Solve back: vR = v + w*L/2, vL = v - w*L/2
		v, w = self.velocity
		vR = v + w * self.wheel_distance / 2.0
		vL = v - w * self.wheel_distance / 2.0
		omega_L = vL / self.wheel_radius
		omega_R = vR / self.wheel_radius
		return omega_L, omega_R
	
	def transform_aruco_to_world(self, aruco_dict):
		result_arr = []
		for aruco_id, positions in aruco_dict.items():
			for marker_pos in positions:
				x_m = float(marker_pos[0, 0])
				y_m = float(marker_pos[1, 0])
				r   = np.hypot(x_m, y_m)
				phi = angle_normalize(np.arctan2(y_m, x_m))
				result_arr.append([r, phi])
		return result_arr
