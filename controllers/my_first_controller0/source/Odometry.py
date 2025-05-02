from controller import Robot
import numpy as np
from controller import Supervisor, Robot

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
		# Init the robot's supervisor
		self.supervisor = Supervisor()
		self.robot_node = self.supervisor.getFromDef("Pioneer 2")
		print(self.robot_node)
		