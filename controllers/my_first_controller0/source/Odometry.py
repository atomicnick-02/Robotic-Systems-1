# Odometry.py

from controller import Robot
import numpy as np

def angle_normalize(angle):
    """Normalize angle to be within -π to π."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class Odometry:
    def __init__(self, robot):
        """
        Initialize Odometry with Webots Robot instance.
        Pose is [x, y, theta], and velocity [v, w].
        """
        # Robot parameters
        self.wheel_radius   = 0.042     # m
        self.wheel_distance = 0.194     # m between wheels
        
        # Time step (s)
        self.dt = robot.getBasicTimeStep() / 1000.0
        
        # State
        self.prev_enc = None            # will hold [left, right] on first update
        self.position = np.zeros(3)     # [x, y, θ]
        self.velocity = np.zeros(2)     # [v (m/s), w (rad/s)]
        
        print("# Odometry initialized successfully #")

    def update_from_encoders(self, left_encoder, right_encoder):
        """
        Read new encoder readings and update pose + velocity.
        
        Returns:
            position: np.array([x, y, θ])
            velocity: np.array([v, w])
        """
        enc = np.array([left_encoder, right_encoder])
        
        # On first call, just store encoders
        if self.prev_enc is None:
            self.prev_enc = enc
            return self.position.copy(), self.velocity.copy()
        
        # Compute wheel travels
        delta = enc - self.prev_enc
        dL = delta[0] * self.wheel_radius
        dR = delta[1] * self.wheel_radius
        
        # Center advance and heading change
        d_center = (dL + dR) / 2.0
        d_theta  = (dR - dL) / self.wheel_distance
        
        # Update pose (use midpoint heading for better accuracy)
        theta_mid = self.position[2] + d_theta / 2.0
        dx = d_center * np.cos(theta_mid)
        dy = d_center * np.sin(theta_mid)
        
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] = angle_normalize(self.position[2] + d_theta)
        
        # Compute velocities
        v = d_center / self.dt
        w = d_theta  / self.dt
        self.velocity = np.array([v, w])
        
        # Save for next iteration
        self.prev_enc = enc
        
        return self.position.copy(), self.velocity.copy()

    def get_velocity(self):
        """
        Returns the last computed velocities [v, w].
        """
        return self.velocity.copy()

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
        print(f"pos: {self.position[2]}")
        theta = self.position[2]
        
        result_arr = []
        
        for aruco_id, positions in aruco_dict.items():
            
            
            for marker_pos in positions:
                # Transform marker position to world coordinates
                # world_pos = marker_pos[:2, :] + robot_position #ignore height
                
                world_pos = marker_pos[:2, :] #relative to robot position 
                # Calculate polar coordinates
                r = float(np.linalg.norm(world_pos.T))
                phi = angle_normalize(np.arctan2(world_pos[1, 0], world_pos[0, 0]) - theta)
                
                result_arr.append([r, float(phi)])
        
        return result_arr