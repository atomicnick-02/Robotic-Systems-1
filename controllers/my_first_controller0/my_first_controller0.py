from controller import Robot
from source.Odometry import Odometry
import numpy as np
from source.ApriltagDetector import AprilTagDetector
from source.EKFSlam import EKF_SLAM

np.set_printoptions(suppress=True, precision=5)

# Coordinate transform for pose
def ekf_to_webots(pose_ekf):
    x_e, y_e, theta_e = pose_ekf
    x_w = y_e
    y_w = -x_e
    theta_w = theta_e - np.pi / 2
    return np.array([x_w, y_w, theta_w])

# Constants
TIME_STEP = 32
MAX_VELOCITY = 26.0
BASE_SPEED = 6.0

COEFFICIENTS = [[15.0, -9.0], [-15.0, 9.0]]

robot = Robot()

# Motors
front_left_motor = robot.getDevice('fl_wheel_joint')
front_right_motor = robot.getDevice('fr_wheel_joint')
rear_left_motor = robot.getDevice('rl_wheel_joint')
rear_right_motor = robot.getDevice('rr_wheel_joint')
for motor in (front_left_motor, front_right_motor, rear_left_motor, rear_right_motor):
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)

# Position sensors
fl_ps = robot.getDevice('front left wheel motor sensor')
fr_ps = robot.getDevice('front right wheel motor sensor')
rl_ps = robot.getDevice('rear left wheel motor sensor')
rr_ps = robot.getDevice('rear right wheel motor sensor')
for ps in (fl_ps, fr_ps, rl_ps, rr_ps):
    ps.enable(TIME_STEP)

# Cameras and lidar
camera_rgb = robot.getDevice('camera')
camera_rgb.enable(TIME_STEP)
lidar = robot.getDevice('laser')
lidar.enable(TIME_STEP)
lidar.enablePointCloud()

# IMU sensors
accelerometer = robot.getDevice('imu accelerometer')
gyro = robot.getDevice('imu gyro')
compass = robot.getDevice('imu compass')
accelerometer.enable(TIME_STEP)
gyro.enable(TIME_STEP)
compass.enable(TIME_STEP)

# Distance sensors
ds_names = ['fl_range', 'rl_range', 'fr_range', 'rr_range']
distance_sensors = []
for name in ds_names:
    ds = robot.getDevice(name)
    ds.enable(TIME_STEP)
    distance_sensors.append(ds)

# Initialize components
odometry = Odometry(robot)
ekf_slam = EKF_SLAM()
april_detector = AprilTagDetector(camera_rgb, robot)

# Main loop
while robot.step(TIME_STEP) != -1:
    # Sensor readings
    distances = [ds.getValue() for ds in distance_sensors]

    image = camera_rgb.getImage()
    enc_vals = [rl_ps.getValue(), rr_ps.getValue()]
    print(f"encoders: {enc_vals[0]:.2f} {enc_vals[1]:.2f}", end=" ")

    # AprilTag detection
    res = april_detector.detect(image)

    # Odometry update
    pose, u = odometry.update_from_encoders(enc_vals[0], enc_vals[1])
    r_phi_dict = odometry.transform_aruco_to_world(res)

    # EKF prediction (velocity in robot frame, no conversion needed)
    ekf_slam.predict(u)

    # EKF correction: flip bearing φ for EKF frame
    if r_phi_dict:
        r_phi_dict_ekf = {
            tag_id: (r, -phi)
            for tag_id, (r, phi) in r_phi_dict.items()
        }
        ekf_slam.correct(r_phi_dict_ekf)

    # Get state and transform for display
    robot_pose, landmarks = ekf_slam.get_state()
    robot_pose_webots = ekf_to_webots(robot_pose)

    print(f"robot pose encoders {pose}")
    try:
        print(f"landmarks: {len(landmarks)}\n{landmarks}")
        error = (pose - robot_pose_webots) / (robot_pose_webots + 1e-6) * 100
        print(error)
        print("-" * 20)
        ekf_slam.plot_landmarks(landmarks, robot_pose)  # Plot still in EKF frame
    except:
        print("No landmarks detected")

    # Collision avoidance
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
