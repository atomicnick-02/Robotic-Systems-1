from controller import Robot, Supervisor, PositionSensor
from source.Odometry import Odometry
import numpy as np
import cv2
from source.ApriltagDetector import AprilTagDetector
np.set_printoptions(suppress=True, precision=4)
# Constants
MAX_SENSOR_NUMBER = 16
RANGE = 1024 / 2

# Helper to clamp values
def bound(x, a, b):
	return a if x < a else b if x > b else x

def initialize(robot) -> dict:
	"""
	Initialize devices, sensors, motors, and Braitenberg weights.
	Returns a dict with all handles and parameters.
	"""
	# Basic timestep
	time_step = int(robot.getBasicTimeStep())
	# Robot name
	robot_name = robot.getName()

	# Default parameters
	camera_enabled = True
	range_val = RANGE
	max_speed = 1.0
	speed_unit = 1.0

	# Weight matrices for various robots
	
	pioneer2_matrix = [
		[-1, 15], [-3, 13], [-3, 8], [-2, 7],
		[-3, -4], [-4, -2], [-3, -2], [-1, -1],
		[-1, -1], [-2, -3], [-2, -4], [4, -3],
		[7, -5], [7, -3], [10, -2], [11, -1]
	]
	

	# Determine robot type and parameters

	if robot_name == "Pioneer 2":
		num_sensors = 16
		sensor_prefix = "ds"
		weights = pioneer2_matrix
		cal_sesnor_prefix = "cal_ds"
		max_speed = 10.0 # Angular speed
		speed_unit = 0.9
	else:
		print("This controller doesn't support robot:", robot_name)
		exit(1)

	# Initialize sensors
	sensors = []
	for i in range(num_sensors):
		name = f"{sensor_prefix}{i}"
		ds = robot.getDevice(name)
		ds.enable(time_step)
		sensors.append(ds)

	# Initialize calibration sensor
	cal_ds = robot.getDevice(f"{cal_sesnor_prefix}")
	cal_ds.enable(time_step)

	# Initialize motors
	left_motor = robot.getDevice("left wheel motor")
	right_motor = robot.getDevice("right wheel motor")
	left_motor.setPosition(float('inf'))
	right_motor.setPosition(float('inf'))
	left_motor.setVelocity(0.0)
	right_motor.setVelocity(0.0)
	# Initialize encoders
	left_encoder = robot.getDevice("left wheel sensor")
	right_encoder = robot.getDevice("right wheel sensor")
	left_encoder.enable(time_step)
	right_encoder.enable(time_step)


	# Initialize camera if present
	cam = None
	if camera_enabled:
		cam = robot.getDevice("camera")
		cam.enable(time_step)
		print(f"Camera enabled with FOV: {cam.getFov()} rad")


	return {
		'time_step': time_step,
		'sensors': sensors,
		'weights': weights,
		'range': range_val,
		'max_speed': max_speed,
		'speed_unit': speed_unit,
		'left_motor': left_motor,
		'right_motor': right_motor,
		'left_encoder': left_encoder,
		'right_encoder': right_encoder,
		'camera': cam,
		'cal_ds': cal_ds,
	}


def run():
	robot = Robot()
	ctx = initialize(robot)
	# Initialize odometry
	odometry = Odometry(robot)
	odometry.update_last_encoder_values() # Create the zero point
	print("Timestep:", ctx['time_step'])
	# np.set_printoptions(suppress = True, precision = 4)
	
	# Initialize AprilTag detector
	if ctx['camera']:
		print("focal length:", ctx['camera'].getFocalLength())
		ctx['AprilTagDetector'] = AprilTagDetector(ctx['camera'], robot)
		print("AprilTag detector initialized.")

		
	print("Starting main loop...")
	while robot.step(ctx['time_step']) != -1:
		
		# SECTION - Measurements	
		image = ctx['camera'].getImage() # Get AprilTags Positions from camera
		# res = ctx['AprilTagDetector'].detect(image)
		# write image to file
		image_array = np.frombuffer(image, np.uint8)
		rgb_image = image_array.reshape((ctx['camera'].getHeight(), ctx['camera'].getWidth(), 4))
		gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2GRAY)
		cv2.imwrite("source/test_april/image3.png", gray_image)
		res = ctx['AprilTagDetector'].detect(image)
		
		print(res)
		odometry.update_last_encoder_values() # Read encoders
		readings = [ds.getValue() for ds in ctx['sensors']] #get distance sensor values
		# !SECTION - Measurements

		# SECTION - Motor Actions
		speed_l = 0.0
		speed_r = 0.0
		for i, val in enumerate(readings):
			factor = 1.0 - (val / ctx['range'])
			speed_l += ctx['speed_unit'] * ctx['weights'][i][0] * factor
			speed_r += ctx['speed_unit'] * ctx['weights'][i][1] * factor
			# speed_l -= 0.5
			# speed_r += 0.5
		# Clamp
		speed_l = bound(speed_l, -ctx['max_speed'], ctx['max_speed'])
		speed_r = bound(speed_r, -ctx['max_speed'], ctx['max_speed'])

		# Set velocities
		ctx['left_motor'].setVelocity(speed_l)
		ctx['right_motor'].setVelocity(speed_r)
		# !SECTION - Motor Actions

		
		# SECTION - After Actions
		# Update the last encoder values



if __name__ == "__main__":
	run()

