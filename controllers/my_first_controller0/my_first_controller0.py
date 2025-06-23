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

# Initialize RGB camera
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

# Initialize odometry with default (0,0,0) start
odometry = Odometry(robot, TIME_STEP)

# Add helper method to convert landmark observations to (x, y)
def transform_aruco_to_world_xy(robot_pose, aruco_dict):
    theta = robot_pose[2]
    x_r, y_r = robot_pose[0], robot_pose[1]
    result_arr = []
    for aruco_id, positions in aruco_dict.items():
        for marker_pos in positions:
            x_m = float(marker_pos[0, 0])
            y_m = float(marker_pos[1, 0])
            x_w = x_r + x_m * np.cos(theta) - y_m * np.sin(theta)
            y_w = y_r + x_m * np.sin(theta) + y_m * np.cos(theta)
            result_arr.append((x_w, y_w))
    return result_arr

# Initialize EKF_SLAM
ekf_slam = EKF_SLAM()
ekf_slam.set_noise_parameters(
    R=np.diag([0.005, 0.005, np.deg2rad(0.2)]) ** 2,
    Q=np.diag([0.1, np.deg2rad(5)]) ** 2
)

# Initialize AprilTag Detector
april_detector = AprilTagDetector(camera_rgb, robot)

# Track whether this is the first update
efk_initialized = False

# Main control loop
while robot.step(TIME_STEP) != -1:
    # Read sensors
    a = accelerometer.getValues()
    distances = [ds.getValue() for ds in distance_sensors]
    image = camera_rgb.getImage()

    # Average encoder readings
    left_enc  = rl_ps.getValue()
    right_enc = rr_ps.getValue()
    enc_vals = [left_enc, right_enc]
    print(f"Encoders: {enc_vals[0]:.2f}, {enc_vals[1]:.2f}", end="  ")

    # Detect AprilTags
    res = april_detector.detect(image)

    # Update odometry and get raw velocity delta u for EKF
    pose, u = odometry.update_from_encoders(enc_vals[0], enc_vals[1])

    # Initialize EKF to odometry pose on first run
    if not efk_initialized:
        ekf_slam.hypotheses[0].mu[:3] = pose.copy()
        efk_initialized = True

    # Log landmark positions in world coordinates (x, y)
    xy_coords = transform_aruco_to_world_xy(pose, res)
    if xy_coords:
        print("Detected landmark positions (x, y):")
        for i, (x, y) in enumerate(xy_coords):
            print(f"  #{i}: x = {x:.3f}, y = {y:.3f}")
    else:
        print("No landmarks detected.")

    # Transform observations for EKF
    r_phi_dict = odometry.transform_aruco_to_world(res)

    # Add Gaussian noise to observations
    Q = ekf_slam.Q
    sigma_r = np.sqrt(Q[0, 0])
    sigma_phi = np.sqrt(Q[1, 1])
    noisy_r_phi_dict = []
    for r, phi in r_phi_dict:
        noisy_r = r + np.random.normal(0, sigma_r)
        noisy_phi = phi + np.random.normal(0, sigma_phi)
        noisy_r_phi_dict.append(np.array([noisy_r, noisy_phi]))

    # EKF-SLAM update using the freshly computed u
    ekf_slam.update(u, noisy_r_phi_dict)
    robot_pose, landmarks = ekf_slam.get_state()

    # Print pose comparisons
    print(f"\nOdometry pose: {pose}")
    print(f"EKF pose:      {robot_pose}")

    try:
        print(f"Landmarks in FOV: {len(r_phi_dict)}")
        print(f"Total landmarks: {len(landmarks)}\n{landmarks}")
    except:
        print("No landmarks detected")

    print("-" * 30)

    # Compute avoidance-based motor speeds
    avoidance = [0.0, 0.0]
    speeds = [0.0, 0.0]
    for i in (0, 1):
        for j in (1, 2):
            delta = 2.0 - distances[j]
            avoidance[i] += delta * delta * COEFFICIENTS[i][j - 1]
        raw_speed = BASE_SPEED + avoidance[i]
        speeds[i] = min(raw_speed, MAX_VELOCITY)

    # Apply motor speeds
    front_left_motor.setVelocity(speeds[0])
    rear_left_motor.setVelocity(speeds[0])
    front_right_motor.setVelocity(speeds[1])
    rear_right_motor.setVelocity(speeds[1])
