from controller import Robot, PositionSensor
import numpy as np


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
		self.robot_pose = np.eye(4)
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
	def cal_linear_vel(self, r_read, l_read) -> tuple:
		"""
		Calculate the wheel speeds based on the encoder values.

		Returns:
			A tuple containing the left and right wheel speeds.
		"""
		left_distance = (l_read - self.enc_tp[0]) 
		right_distance = (r_read - self.enc_tp[1]) 
		self.enc_tp = (l_read, r_read) # Update the last encoder values
		
		l_lin_vel = round(left_distance / self.dt * 1000, 6) * self.wheel_radius
		r_lin_vel = round(right_distance / self.dt *1000, 6) * self.wheel_radius

		dtheta = (r_lin_vel - l_lin_vel) / self.wheel_distance
		dxy = (r_lin_vel + l_lin_vel) / 2
		dx = dxy * np.cos(dtheta)
		dy = dxy * np.sin(dtheta)
		
		dpos = np.array([dx, dy, dtheta]).reshape(3, 1)
		self.position = self.position + dpos
		print("Position:", self.position.T)

		print("dpos:", dpos.T)
		
		return self.position


	def cal_arouco_to_world(self, aruco_dict: dict) -> dict:
		"""
		Calculate the position of the Aruco tags in world coordinates.

		Args:
			aruco_dict: A dictionary containing the Aruco tag IDs and their positions.

		Returns:
			A dictionary containing the Aruco tag IDs and r,phi.
		"""
		for key in aruco_dict:
			# if the item has more than 1 values
			print(f"key: {key}")
			if len(aruco_dict[key]) > 0:
				for item in aruco_dict[key]:
					print(item)
		return aruco_dict
