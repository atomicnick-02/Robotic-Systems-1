import numpy as np

class DifferentialDriveRobot:
    def __init__(self, x=0, y=0, theta=0, wheel_radius=0.05, wheel_base=0.2):
        # Robot state: [x, y, theta]
        self.x = x
        self.y = y
        self.theta = theta
        
        # Robot parameters
        self.wheel_radius = wheel_radius  # m
        self.wheel_base = wheel_base  # m
        
        # Control inputs: [v_l, v_r] (left and right wheel velocities)
        self.v_l = 0
        self.v_r = 0
       
        # Robot history for plotting path
        self.history = {'x': [x], 'y': [y], 'theta': [theta]}
        
        # Noise parameters
        self.motion_noise = 0.01  # std dev
        self.measurement_noise = 0.1  # std dev
        
    def move(self, dt, v_l, v_r):
        """Move the robot according to differential drive kinematics"""
        # Linear and angular velocities
        v = (v_r + v_l) / 2.0
        omega = (v_r - v_l) / self.wheel_base
        
        # Add noise to motion
        v_noisy = v + np.random.normal(0, self.motion_noise)
        omega_noisy = omega + np.random.normal(0, self.motion_noise)
        
        # Update state
        if abs(omega_noisy) < 1e-6:  # Moving in a straight line
            self.x += v_noisy * dt * np.cos(self.theta)
            self.y += v_noisy * dt * np.sin(self.theta)
        else:  # Moving in an arc
            radius = v_noisy / omega_noisy
            self.x += radius * (np.sin(self.theta + omega_noisy * dt) - np.sin(self.theta))
            self.y += radius * (-np.cos(self.theta + omega_noisy * dt) + np.cos(self.theta))
            self.theta += omega_noisy * dt
            
        # Normalize theta to [-pi, pi]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
        # Record history
        self.history['x'].append(self.x)
        self.history['y'].append(self.y)
        self.history['theta'].append(self.theta)
        
    def set_wheel_velocities(self, v_l, v_r):
        """Set wheel velocities"""
        self.v_l = v_l
        self.v_r = v_r
        
    def measure_landmark(self, landmark_x, landmark_y):
        """Measure range and bearing to landmark with noise"""
        dx = landmark_x - self.x
        dy = landmark_y - self.y
        
        # Range and bearing
        range_to_landmark = np.sqrt(dx**2 + dy**2)
        bearing = np.arctan2(dy, dx) - self.theta
        
        # Normalize bearing to [-pi, pi]
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        
        # Add measurement noise
        range_with_noise = range_to_landmark + np.random.normal(0, self.measurement_noise)
        bearing_with_noise = bearing + np.random.normal(0, self.measurement_noise)
        
        # Check if landmark is within sensor range
        if range_to_landmark <= self.sensor_range:
            return range_with_noise, bearing_with_noise
        else:
            return None, None
        
    def get_pose(self):
        """Return the current pose"""
        return np.array([self.x, self.y, self.theta])
