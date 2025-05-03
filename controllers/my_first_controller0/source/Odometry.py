from controller import Robot
import numpy as np


class Odometry:
	def __init__(self, robot: Robot):
		"""
		Initialize the Odometry class with a robot instance.
		This class is only responsible for odometry calculations
		and does not output any actions to the robot.

		Args:
			robot: An instance of the Robot class from the Webots API.
		"""
		self.robot = robot
		# Initialize robot parameters
		self.time_step = int(robot.getBasicTimeStep())
		self.left_encoder = robot.getDevice("left wheel sensor")
		self.right_encoder = robot.getDevice("right wheel sensor")
		# Initialize encoders
		self.left_encoder.enable(self.time_step)
		self.right_encoder.enable(self.time_step)
		# Init the robot's wheel radius and distance between wheels
		self.wheel_radius = 0.0825  # in meters
		self.wheel_distance = 0.331
		# The last encoder values
		self.last_left_encoder_value = 0.0
		self.last_right_encoder_value = 0.0
		
	def read_encoders(self) -> tuple:
		"""
		Read the encoder values for the left and right wheels.

		Returns:
			A tuple containing the left and right encoder values.
		"""
		left_encoder_value = round(self.left_encoder.getValue(), 2)
		right_encoder_value = round(self.right_encoder.getValue(),2)
		
		
		return left_encoder_value, right_encoder_value
	
	def calculate_distance(self) -> tuple:
		"""
		Calculate the distance traveled by the left and right wheels
		since the last encoder reading.

		Returns:
			A tuple containing the distance traveled by the left and right wheels.
		"""
		left_encoder_value, right_encoder_value = self.read_encoders()
		
		left_distance = (left_encoder_value - self.last_left_encoder_value) * self.wheel_radius
		right_distance = (right_encoder_value - self.last_right_encoder_value) * self.wheel_radius
		
		return left_distance, right_distance
