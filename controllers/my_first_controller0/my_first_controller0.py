from controller import Robot, Node
import numpy as np
import cv2
import apriltag
from ApriltagDetector import AprilTagDetector

# Constants
MAX_SENSOR_NUMBER = 16
RANGE = 1024 / 2

# Helper to clamp values
def bound(x, a, b):
    return a if x < a else b if x > b else x


def initialize(robot):
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
        calibration_sensor = 1
        cal_sesnor_prefix = "cal_ds"
        max_speed = 3.0
        speed_unit = 0.3
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

    # Initialize camera if present
    cam = None
    if camera_enabled:
        cam = robot.getDevice("camera")
        cam.enable(time_step)
        # cam.recognitionEnable(time_step)
        print(f"Camera enabled with FOV: {cam.getFov()} rad")


    print(f"Initialized {robot_name} with {num_sensors} sensors.")

    return {
        'time_step': time_step,
        'sensors': sensors,
        'weights': weights,
        'range': range_val,
        'max_speed': max_speed,
        'speed_unit': speed_unit,
        'left_motor': left_motor,
        'right_motor': right_motor,
        'camera': cam,

        'cal_ds': cal_ds,
    }


def run():
    robot = Robot()
    ctx = initialize(robot)
    # Initialize AprilTag detector
    if ctx['camera']:
        print("focal length:", ctx['camera'].getFocalLength())
        ctx['AprilTagDetector'] = AprilTagDetector(ctx['camera'], robot)
        print("AprilTag detector initialized.")
    # cal_ds = ctx['cal_ds']
    # print all keys in ctx
    
    while robot.step(ctx['time_step']) != -1:
        # Refresh camera image
    
        # Detect AprilTags
        if ctx['camera']:
            # Get the image from the camera
            image = ctx['camera'].getImage()
            ctx['AprilTagDetector'].detect(image)
            
        #read the calibration sensor

        cal_ds_value = ctx['cal_ds'].getValue()
        print("Calibration sensor value:", cal_ds_value)
        # Read sensor values
        readings = [ds.getValue() for ds in ctx['sensors']]
        # print the value of ds0
        # print("Sensor values:", readings)
        # Braitenberg: compute wheel speeds
        speed_l = 0.0
        speed_r = 0.0
        for i, val in enumerate(readings):
            factor = 1.0 - (val / ctx['range'])
            speed_l += ctx['speed_unit'] * ctx['weights'][i][0] * factor
            speed_r += ctx['speed_unit'] * ctx['weights'][i][1] * factor

        # Clamp
        speed_l = bound(speed_l, -ctx['max_speed'], ctx['max_speed'])
        speed_r = bound(speed_r, -ctx['max_speed'], ctx['max_speed'])

        # Set velocities
        ctx['left_motor'].setVelocity(speed_l)
        ctx['right_motor'].setVelocity(speed_r)
        


if __name__ == "__main__":
    run()
