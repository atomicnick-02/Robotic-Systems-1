from controller import Robot, PositionSensor
import numpy as np

def angle_normalize(angle):
    """Normalize angle to be within -π to π range."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi

class Odometry:
    def __init__(self, robot):
        """
        Initialize the Odometry class with a robot instance.
        This class is responsible for odometry calculations.

        Args:
            robot: An instance of the Robot class from the Webots API.
        """
        # Robot physical parameters
        self.wheel_radius = 0.0825  # meters
        self.wheel_distance = 0.331  # meters
        
        # Time step
        self.dt = robot.getBasicTimeStep() / 1000  # convert to seconds
        
        # Tracking variables
        self.prev_encoder_values = np.zeros(2)  # [left, right]
        self.position = np.zeros((3, 1))  # [theta, x, y]
        
        print("# Odometry initialized successfully #")
    
    def update_from_encoders(self, left_encoder, right_encoder):
        """
        Update robot position based on wheel encoder readings.
        
        Args:
            left_encoder: Current left wheel encoder value
            right_encoder: Current right wheel encoder value
            
        Returns:
            Updated position [theta, x, y]
        """
        # Calculate encoder differences
        encoder_diff = np.array([left_encoder, right_encoder]) - self.prev_encoder_values
        
        # Calculate wheel distances
        wheel_distances = encoder_diff * self.wheel_radius
        
        # Calculate robot velocity components
        v = (wheel_distances[1] + wheel_distances[0]) / 2
        w = (wheel_distances[1] - wheel_distances[0]) / self.wheel_distance
        
        # Store velocities for possible external use
        self.velocity = np.array([[v], [w]])
        
        # Update position using current orientation
        theta = self.position[0, 0]
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = w
        
        # Apply position update
        self.position += np.array([[dtheta], [dx], [dy]])
        
        # Normalize theta to keep it within -π to π
        self.position[0, 0] = angle_normalize(self.position[0, 0])
        
        # Update previous encoder values
        self.prev_encoder_values = np.array([left_encoder, right_encoder])
        
        return self.position
    
    def get_wheel_velocities(self, left_encoder, right_encoder):
        """
        Calculate the angular velocities of both wheels.
        
        Args:
            left_encoder: Current left wheel encoder value
            right_encoder: Current right wheel encoder value
            
        Returns:
            Tuple of (left_angular_velocity, right_angular_velocity)
        """
        encoder_diff = np.array([left_encoder, right_encoder]) - self.prev_encoder_values
        
        # Calculate angular velocities
        angular_velocities = encoder_diff / self.dt
        
        # Update previous encoder values
        self.prev_encoder_values = np.array([left_encoder, right_encoder])
        
        return tuple(angular_velocities)
    
    def transform_aruco_to_world(self, aruco_dict):
        """
        Transform Aruco marker positions from robot frame to world coordinates.
        
        Args:
            aruco_dict: Dictionary with Aruco IDs as keys and marker positions as values
            
        Returns:
            Dictionary of Aruco IDs with corresponding (r, phi) polar coordinates
        """
        theta = self.position[0, 0]
        robot_position = self.position[1:, :]
        
        result_dict = {}
        
        for aruco_id, positions in aruco_dict.items():
            if not positions:  # Skip empty lists
                continue
                
            result_dict[aruco_id] = []
            
            for marker_pos in positions:
                # Transform marker position to world coordinates
                world_pos = marker_pos[:2, :] + robot_position
                
                # Calculate polar coordinates
                r = float(np.linalg.norm(world_pos.T))
                phi = angle_normalize(np.arctan2(world_pos[1, 0], world_pos[0, 0]) - theta)
                
                result_dict[aruco_id].append((r, float(phi)))
        
        return result_dict