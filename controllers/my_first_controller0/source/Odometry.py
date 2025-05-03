from controller import Robot, PositionSensor
import numpy as np


class Odometry:
	def __init__(self, robot: Robot, ctx: dict = None):
		"""
		Initialize the Odometry class with a robot instance.
		This class is only responsible for odometry calculations
		and does not output any actions to the robot.

		Args:
			robot: An instance of the Robot class from the Webots API.
		"""
		self.robot = robot
		# Initialize robot parameters
		self.dt = int(robot.getBasicTimeStep())
		self.l_encoder = ctx['left_encoder'] if ctx else robot.getDevice("left wheel sensor")
		self.r_encoder = ctx['right_encoder'] if ctx else robot.getDevice("right wheel sensor")

		# Initialize encoders
		self.l_encoder.enable(self.dt)
		self.r_encoder.enable(self.dt)
		# Init the robot's wheel radius and distance between wheels
		self.wheel_radius = 0.0825  # in meters
		self.wheel_distance = 0.331
		# The last encoder values
		self.enc_tp = (0, 0) # Left and right encoder values past
		self.enc_tc = (0, 0) # Left and right encoder values current


	def update_last_encoder_values(self):
		"""
		Update the last encoder values with the current values.
		"""
		# self.enc_tp = self.enc_tc
		self.enc_tc = self.read_encoders() 
	
	def read_encoders(self) -> tuple:
		"""
		Read the current values of the left and right encoders.

		Returns:
			A tuple containing the left and right encoder values.
		"""
		l_encoder_value = self.l_encoder.getValue()
		r_encoder_value = self.r_encoder.getValue()
		
		
		return l_encoder_value, r_encoder_value
	
	def calculate_distance(self) -> tuple:
		"""
		Calculate the distance traveled by the left and right wheels
		since the last encoder reading.

		Returns:
			A tuple containing the distance traveled by the left and right wheels.
		"""
		l_encoder_value, r_encoder_value = self.read_encoders()
		
		left_distance = (l_encoder_value - self.enc_tp[0]) * self.wheel_radius
		right_distance = (r_encoder_value - self.enc_tp[1]) * self.wheel_radius

		return left_distance, right_distance
	def calculate_angular_vel(self) -> tuple:
		"""
		Calculate the angular velocity of the robot based on the wheel speeds.

		Returns:
			A tuple containing the left and right angular velocities.
		"""
		l_enc_diff = self.enc_tc[0] - self.enc_tp[0]
		r_enc_diff = self.enc_tc[1] - self.enc_tp[1]
		left_distance = l_enc_diff * self.wheel_radius
		right_distance = r_enc_diff * self.wheel_radius
		
		left_angular_velocity = left_distance / self.dt
		right_angular_velocity = right_distance / self.dt
		
		return left_angular_velocity, right_angular_velocity
	def calculate_linear_vel(self) -> tuple:
		"""
		Calculate the wheel speeds based on the encoder values.

		Returns:
			A tuple containing the left and right wheel speeds.
		"""
		left_distance, right_distance = self.calculate_distance()
		
		left_speed = left_distance / self.dt
		right_speed = right_distance / self.dt
		
		return left_speed, right_speed

