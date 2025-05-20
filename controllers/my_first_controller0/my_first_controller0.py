from controller import Robot
from source.Odometry import Odometry
import numpy as np
from source.ApriltagDetector import AprilTagDetector
from source.EKFSlam  import EKF_SLAM

np.set_printoptions(suppress=True, precision=5)

# Constants
TIME_STEP = 32
MAX_VELOCITY = 26.0
BASE_SPEED = 6.0

# Empirical collision avoidance coefficients
COEFFICIENTS = [[15.0, -9.0],
				[-15.0, 9.0]]

# Initialize the robot instance
robot = Robot()

# Initialize motors
front_left_motor = robot.getDevice('fl_wheel_joint')
front_right_motor = robot.getDevice('fr_wheel_joint')
rear_left_motor = robot.getDevice('rl_wheel_joint')
rear_right_motor = robot.getDevice('rr_wheel_joint')
for motor in (front_left_motor, front_right_motor, rear_left_motor, rear_right_motor):
	motor.setPosition(float('inf'))
	motor.setVelocity(0.0)

# Initialize position sensors
fl_ps = robot.getDevice('front left wheel motor sensor')
fr_ps = robot.getDevice('front right wheel motor sensor')
rl_ps = robot.getDevice('rear left wheel motor sensor')
rr_ps = robot.getDevice('rear right wheel motor sensor')
for ps in (fl_ps, fr_ps, rl_ps, rr_ps):
	ps.enable(TIME_STEP)

# Initialize RGB and depth cameras
camera_rgb = robot.getDevice('camera')
camera_rgb.enable(TIME_STEP)

# Initialize lidar
lidar = robot.getDevice('laser')
lidar.enable(TIME_STEP)
lidar.enablePointCloud()

# Initialize IMU sensors
accelerometer = robot.getDevice('imu accelerometer')
gyro = robot.getDevice('imu gyro')
compass = robot.getDevice('imu compass')
accelerometer.enable(TIME_STEP)
gyro.enable(TIME_STEP)
compass.enable(TIME_STEP)

# Initialize distance sensors
ds_names = ['fl_range', 'rl_range', 'fr_range', 'rr_range']
distance_sensors = []
for name in ds_names:
	ds = robot.getDevice(name)
	ds.enable(TIME_STEP)
	distance_sensors.append(ds)
	

# Initialize odometry
odometry = Odometry(robot)

# Initialize EKF_SLAM
ekf_slam = EKF_SLAM()

# Initialize AprilTag Detector
april_detector = AprilTagDetector(camera_rgb,robot)

# Main control loop
while robot.step(TIME_STEP) != -1:
	# Read accelerometer
	a = accelerometer.getValues()
	# print(f"accelerometer values = {a[0]:.2f} {a[1]:.2f} {a[2]:.2f}", end= " ")

	# Read distance sensor values
	distances = [ds.getValue() for ds in distance_sensors]

	#SECTION -  Read Camera and position from wheels
	image = camera_rgb.getImage() # Get AprilTags Positions from camera
	enc_vals = [rl_ps.getValue(), rr_ps.getValue()]
	# print(f"encoders: {enc_vals[0]:.2f} {enc_vals[1]:.2f}", end= " ")
	res = april_detector.detect(image)
	pose, u = odometry.update_from_encoders(enc_vals[0], enc_vals[1])
	r_phi_dict = odometry.transform_aruco_to_world(res)
	
	u = odometry.get_velocity()
	ekf_slam.predict(u)
	if r_phi_dict:
		ekf_slam.correct(r_phi_dict)
	
	robot_pose, landmarks = ekf_slam.get_state()
	print(f"robot pose encoders {pose}, ekf: {robot_pose}")
	# try:
	# 	print(f"landmarks: {len(landmarks)}\n{landmarks}")
	# 	error = (pose - robot_pose)/robot_pose *100
	# 	print(error)
		
	# 	print("-"*20)
	# 	# ekf_slam.plot_landmarks(landmarks, robot_pose)
	# except:
		# print("No landmarks detected")

	# image_array = np.frombuffer(image, np.uint8)
	# rgb_image = image_array.reshape((camera_rgb.getHeight(), camera_rgb.getWidth(), 4))
	# gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2GRAY)
	# cv2.imshow('image', rgb_image)
	# cv2.waitKey(1)
	# timestep = str(robot.getTime())
	# print(timestep[2:])
	# cv2.imwrite(f'img{timestep[2:]}.png', gray)


	# Compute avoidance-based motor speeds
	avoidance = [0.0, 0.0]
	speeds = [0.0, 0.0]
	for i in (0, 1):
		for j in (1, 2):  # front-left and front-right sensors
			delta = 2.0 - distances[j]
			avoidance[i] += delta * delta * COEFFICIENTS[i][j-1]
		raw_speed = BASE_SPEED + avoidance[i]
		speeds[i] = min(raw_speed, MAX_VELOCITY)

	# Apply speeds
	front_left_motor.setVelocity(speeds[0])
	rear_left_motor.setVelocity(speeds[0])
	front_right_motor.setVelocity(speeds[1])
	rear_right_motor.setVelocity(speeds[1])
