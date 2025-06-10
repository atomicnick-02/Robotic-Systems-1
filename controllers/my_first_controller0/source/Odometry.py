from controller import Robot
import numpy as np

def angle_normalize(angle):
	"""Normalize angle to be within -π to π."""

	return (angle + np.pi) % (2 * np.pi) - np.pi



class Odometry:
	def __init__(self, robot):
		"""
		Initialize Odometry with Webots Robot instance.
		Pose is [x, y, theta], and velocity [v, w].
		"""
		# Robot parameters
		self.wheel_radius   = 0.043     # m
		self.wheel_distance = 0.195     # m between wheels
		
		# Time step (s)
		self.dt = robot.getBasicTimeStep() / 1000.0
		
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
		theta_mid = theta_old + d_theta / 2.0

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

		return self.position.copy(), self.velocity.copy()

	def get_velocity(self):
		"""
		Returns the last computed velocities [v, w].
		"""
		return self.velocity.copy()

	def get_wheel_velocities(self, left_encoder, right_encoder):
		"""
		Calculate the angular velocities of both wheels.
		
		Args:
			left_encoder: Current left wheel encoder value
			right_encoder: Current right wheel encoder value
			
		Returns:
			Tuple of (left_angular_velocity, right_angular_velocity)
		"""
		if self.prev_enc is None:
			self.prev_enc = np.array([left_encoder, right_encoder])
			return 0.0, 0.0
		encoder_diff = np.array([left_encoder, right_encoder]) - self.prev_enc
		
		# Calculate angular velocities
		angular_velocities = encoder_diff / self.dt
		
		# Update previous encoder values
		self.prev_enc = np.array([left_encoder, right_encoder])
		
		return tuple(angular_velocities)
	
	def transform_aruco_to_world(self, aruco_dict):
		"""
		Transform Aruco marker positions from robot frame to world polar coordinates.

		Args:
			aruco_dict: Dictionary with Aruco IDs as keys and marker positions (list of np.array) as values

		Returns:
			List of [r, phi] tuples for detected markers
		"""
		theta = self.position[2]
		x_r, y_r = self.position[0], self.position[1]

		result_arr = []

		for aruco_id, positions in aruco_dict.items():
			for marker_pos in positions:
				# Marker position in robot frame
				x_m = float(marker_pos[0, 0])
				y_m = float(marker_pos[1, 0])

				# Convert to world coordinates
				x_w = x_r + x_m * np.cos(theta) - y_m * np.sin(theta)
				y_w = y_r + x_m * np.sin(theta) + y_m * np.cos(theta)

				# Convert back to polar relative to robot pose
				dx = x_w - x_r
				dy = y_w - y_r
				r = np.hypot(dx, dy)
				phi = angle_normalize(np.arctan2(dy, dx) - theta)

				result_arr.append([r, phi])

		return result_arr
