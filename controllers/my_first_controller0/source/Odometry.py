from controller import Robot, PositionSensor
import numpy as np

def angle_dist(b, a):
    theta = b - a
    while theta < -np.pi:
        theta += 2. * np.pi
    while theta > np.pi:
        theta -= 2. * np.pi
    return theta


class Odometry:
	def __init__(self, robot):
		"""
		Initialize the Odometry class with a robot instance.
		This class is only responsible for odometry calculations
		and does not output any actions to the robot.

		Args:
			robot: An instance of the Robot class from the Webots API.
		"""
		# Init the robot's wheel radius and distance between wheels
		self.wheel_radius = 0.0825  # in meters
		self.wheel_distance = 0.331
		# The last encoder values
		self.enc_tp = (0., 0.) # Left and right encoder values past
		# Robot World Position
		self.position = np.array([0., 0., 0.]).reshape(3, 1) # theta, x ,y
		self.dt = robot.getBasicTimeStep() / 1000
		print("#                     Odometry initialized.                      #")

	def cal_distance(self, r_read, l_read) -> tuple:
		"""
		Calculate the distance traveled by the left and right wheels
		since the last encoder reading.

		Returns:
			A tuple containing the distance traveled by the left and right wheels.
		"""
		left_distance = (l_read) * self.wheel_radius
		right_distance = (r_read) * self.wheel_radius
		self.enc_tp = (l_read, r_read) # Update the last encoder values
		return left_distance, right_distance
	
	def cal_angular_vel(self, r_read, l_read) -> tuple:
		"""
		Calculate the angular velocity of the robot based on the wheel speeds.

		Returns:
			A tuple containing the left and right angular velocities.
		"""
		left_distance = (l_read - self.enc_tp[0]) 
		right_distance = (r_read - self.enc_tp[1]) 
		self.enc_tp = (l_read, r_read) # Update the last encoder values
		
		left_angular_velocity = round(left_distance / self.dt , 6) # I am dealing with milliseconds
		right_angular_velocity = round(right_distance / self.dt , 6)
		
		return left_angular_velocity, right_angular_velocity
	def cal_Vb(self, r_read, l_read) -> tuple:
		"""
		Calculate the wheel speeds based on the encoder values.

		Returns:
			A tuple containing the left and right wheel speeds.
		"""
		left_distance = (l_read - self.enc_tp[0]) 
		right_distance = (r_read - self.enc_tp[1]) 
		self.enc_tp = (l_read, r_read) # Update the last encoder values
		
		l_lin_vel = round(left_distance  , 6) * self.wheel_radius
		r_lin_vel = round(right_distance , 6) * self.wheel_radius

		w = (r_lin_vel - l_lin_vel) / self.wheel_distance
		V = (r_lin_vel + l_lin_vel) / 2	
		
		return np.array([[V, w]]).T

	def update_pose(self, position:np.array, V:np.array):
		"""
		Update the robot position given its previous and its Velocity tuple
		"""
		u = V[0]
		w = V[1]
		dx = u * np.cos(w)
		dy = u * np.sin(w)
		dpos = np.array([w, dx, dy]).T

		position = position + dpos.T
		return position
	def cal_arouco_to_world(self, robot_state:np.array, aruco_dict: dict) -> dict:
		"""
		Calculate the position of the Aruco tags in world coordinates.

		Args:
			aruco_dict: A dictionary containing the Aruco tag IDs and their positions.

		Returns:
			A dictionary containing the Aruco tag IDs and r,phi.
		"""
		print(f"robot pose: {robot_state.T}")
		robot_pose = robot_state[1:,:]
		theta = robot_state[0]
		result_dict = {} #store the Id: r,phi
		for key in aruco_dict:
			# if the item has more than 1 values
			print(f"key: {key}")
			if len(aruco_dict[key]) > 0:

				for item in aruco_dict[key]:
					temp_res = item[:2,:] + robot_pose
					# print(temp_res.T)
					r = float(np.linalg.norm(temp_res.T))
					phi = np.arctan2(temp_res[1], temp_res[0]) - theta
					phi = float(angle_dist(phi,0))
					if key not in result_dict:
						result_dict[key] = []
					result_dict[key].append((r, phi))
					
		return result_dict
