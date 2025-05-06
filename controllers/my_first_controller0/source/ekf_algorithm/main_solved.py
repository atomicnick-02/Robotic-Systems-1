import numpy as np
import pygame
import matplotlib.pyplot as plt
import math
import time
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

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
        
        # Maximum velocities
        self.max_wheel_velocity = 2.0  # m/s
        
        # Robot sensor range
        self.sensor_range = 2.0  # meters
        
        # Robot history for plotting path
        self.history = {'x': [x], 'y': [y], 'theta': [theta]}
        
        # Noise parameters
        self.motion_noise = 0.01  # std dev
        self.measurement_noise = 0.1  # std dev
        
    def move(self, dt):
        """Move the robot according to differential drive kinematics"""
        # Linear and angular velocities
        v = (self.v_r + self.v_l) / 2.0
        omega = (self.v_r - self.v_l) / self.wheel_base
        
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
        self.v_l = np.clip(v_l, -self.max_wheel_velocity, self.max_wheel_velocity)
        self.v_r = np.clip(v_r, -self.max_wheel_velocity, self.max_wheel_velocity)
        
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

class EKF_SLAM:
    def __init__(self, robot):
        # Reference to the robot
        self.robot = robot
        
        # Initial state vector [x, y, theta]
        self.mu = np.array([[robot.x], [robot.y], [robot.theta]])
        
        # Initial covariance matrix
        self.sigma = np.eye(3) * 0.01
        
        # Landmark positions (will be expanded as landmarks are observed)
        self.landmarks = {}  # Dictionary mapping landmark IDs to their indices in state vector
        
        # Process and measurement noise
        self.R = np.diag([0.01, 0.01, 0.01])  # Process noise
        self.Q = np.diag([0.1, 0.1])  # Measurement noise
        
        # To visualize uncertainties
        self.estimated_landmarks = []  # List of (x, y, uncertainty) for visualization
        
    def predict(self, dt):
        """EKF prediction step"""
        # Current state
        x, y, theta = self.mu[0, 0], self.mu[1, 0], self.mu[2, 0]
        
        # Control inputs (from robot's wheel velocities)
        v = (self.robot.v_r + self.robot.v_l) / 2.0
        omega = (self.robot.v_r - self.robot.v_l) / self.robot.wheel_base
        
        # Predict next state
        if abs(omega) < 1e-6:  # Moving in a straight line
            x_next = x + v * dt * np.cos(theta)
            y_next = y + v * dt * np.sin(theta)
            theta_next = theta
        else:  # Moving in an arc
            x_next = x + (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            y_next = y + (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))
            theta_next = theta + omega * dt
        
        # Update state vector for robot pose
        self.mu[0, 0] = x_next
        self.mu[1, 0] = y_next
        self.mu[2, 0] = np.arctan2(np.sin(theta_next), np.cos(theta_next))
        
        # Compute Jacobian G of motion model
        G = np.eye(len(self.mu))
        if abs(omega) < 1e-6:  # Moving in a straight line
            G[0, 2] = -v * dt * np.sin(theta)
            G[1, 2] = v * dt * np.cos(theta)
        else:  # Moving in an arc
            G[0, 2] = (v / omega) * (np.cos(theta + omega * dt) - np.cos(theta))
            G[1, 2] = (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
        
        # Update covariance matrix
        R_full = np.zeros((len(self.mu), len(self.mu)))
        R_full[:3, :3] = self.R  # Only apply process noise to robot state
        
        self.sigma = G @ self.sigma @ G.T + R_full
        
        # Update estimated landmark positions for visualization
        self.update_landmark_estimates()
        
    def update(self, landmark_id, measurement):
        """EKF update step for a landmark observation"""
        # Extract measurement
        r, phi = measurement
        
        # If this is a new landmark
        if landmark_id not in self.landmarks:
            # Initialize landmark position based on measurement
            lx = self.mu[0, 0] + r * np.cos(phi + self.mu[2, 0])
            ly = self.mu[1, 0] + r * np.sin(phi + self.mu[2, 0])
            
            # Add landmark to state vector
            self.mu = np.vstack((self.mu, [[lx], [ly]]))
            
            # Calculate the index in the state vector
            landmark_index = (len(self.mu) - 2) // 2
            
            # Update landmark dictionary
            self.landmarks[landmark_id] = landmark_index
            
            # Expand covariance matrix
            old_size = self.sigma.shape[0]
            new_sigma = np.zeros((old_size + 2, old_size + 2))
            new_sigma[:old_size, :old_size] = self.sigma
            
            # Initialize landmark covariance with high uncertainty
            new_sigma[old_size:, old_size:] = np.eye(2) * 1000
            
            # Set up cross-correlations
            # This could be improved with proper calculation
            self.sigma = new_sigma
            
        # Get landmark index
        landmark_index = self.landmarks[landmark_id]
        
        # Calculate the position in the state vector
        pos = 3 + 2 * (landmark_index - 1) if landmark_index > 0 else 3
        
        # Compute expected measurement
        lx, ly = self.mu[pos, 0], self.mu[pos + 1, 0]
        dx = lx - self.mu[0, 0]
        dy = ly - self.mu[1, 0]
        
        q = dx**2 + dy**2
        
        # Avoid division by zero
        if q < 1e-6:
            q = 1e-6
            
        r_expected = np.sqrt(q)
        phi_expected = np.arctan2(dy, dx) - self.mu[2, 0]
        phi_expected = np.arctan2(np.sin(phi_expected), np.cos(phi_expected))
        
        z_expected = np.array([[r_expected], [phi_expected]])
        z_actual = np.array([[r], [phi]])
        
        # Compute innovation
        innovation = z_actual - z_expected
        innovation[1, 0] = np.arctan2(np.sin(innovation[1, 0]), np.cos(innovation[1, 0]))
        
        # Compute Jacobian H
        H = self.compute_measurement_jacobian(landmark_index)
        
        # Compute innovation covariance
        S = H @ self.sigma @ H.T + self.Q
        
        # Compute Kalman gain
        K = self.sigma @ H.T @ np.linalg.inv(S)
        
        # Update state vector
        self.mu = self.mu + K @ innovation
        
        # Update covariance matrix
        self.sigma = (np.eye(len(self.mu)) - K @ H) @ self.sigma
        
        # Update estimated landmark positions for visualization
        self.update_landmark_estimates()
    
    def compute_measurement_jacobian(self, landmark_index):
        """Compute the Jacobian of the measurement model for a specific landmark"""
        # Calculate the position in the state vector
        pos = 3 + 2 * (landmark_index - 1) if landmark_index > 0 else 3
        
        # Get robot and landmark positions
        x, y, theta = self.mu[0, 0], self.mu[1, 0], self.mu[2, 0]
        lx, ly = self.mu[pos, 0], self.mu[pos + 1, 0]
        
        # Compute differences and squared distance
        dx = lx - x
        dy = ly - y
        q = dx**2 + dy**2
        
        # Check for singularity
        if q < 1e-6:
            q = 1e-6
        
        # Initialize Jacobian with zeros
        H = np.zeros((2, len(self.mu)))
        
        # Fill in the Jacobian
        sqrt_q = np.sqrt(q)
        
        # Derivatives of range measurement
        H[0, 0] = -dx / sqrt_q
        H[0, 1] = -dy / sqrt_q
        H[0, 2] = 0
        H[0, pos] = dx / sqrt_q
        H[0, pos + 1] = dy / sqrt_q
        
        # Derivatives of bearing measurement
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1
        H[1, pos] = -dy / q
        H[1, pos + 1] = dx / q
        
        return H
    
    def update_landmark_estimates(self):
        """Update the list of estimated landmarks with their uncertainties"""
        self.estimated_landmarks = []
        for landmark_id, landmark_index in self.landmarks.items():
            # Calculate the position in the state vector
            pos = 3 + 2 * (landmark_index - 1) if landmark_index > 0 else 3
            
            # Get landmark position
            lx = self.mu[pos, 0]
            ly = self.mu[pos + 1, 0]
            
            # Get uncertainty (trace of covariance submatrix)
            uncertainty = self.sigma[pos, pos] + self.sigma[pos + 1, pos + 1]
            
            self.estimated_landmarks.append((lx, ly, uncertainty))

class Simulator:
    def __init__(self, width=800, height=600):
        # Initialize pygame
        pygame.init()
        
        # Screen parameters
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("EKF SLAM Simulator")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)
        self.PURPLE = (128, 0, 128)
        
        # Scale from world coordinates to screen coordinates
        self.scale = 100  # pixels per meter
        
        # Robot
        self.robot = DifferentialDriveRobot(x=width/(2*self.scale), y=height/(2*self.scale))
        
        # Initialize EKF SLAM
        self.ekf = EKF_SLAM(self.robot)
        
        # Landmarks (can be added later)
        self.landmarks = []  # List of (x, y) tuples
        
        # Font for displaying information
        self.font = pygame.font.Font(None, 24)
        
        # Clock for timing
        self.clock = pygame.time.Clock()
        
        # Running flag
        self.running = True
        
        # Show grid
        self.show_grid = True
        
        # Show sensor range
        self.show_sensor_range = True
        
        # Show EKF estimates
        self.show_ekf = True
        
        # Debug info
        self.last_observations = []  # List of (landmark_id, range, bearing) tuples
        
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates"""
        screen_x = int(x * self.scale)
        screen_y = int(self.height - y * self.scale)  # Flip y-axis
        return screen_x, screen_y
        
    def run(self):
        # Main game loop
        while self.running:
            # Keep loop running at the right speed
            dt = self.clock.tick(60) / 1000.0  # Convert to seconds
            
            # Process input/events
            self.process_events()
            
            # Update robot position
            self.robot.move(dt)
            
            # Update EKF prediction
            self.ekf.predict(dt)
            
            # Process landmark observations
            self.process_observations()
            
            # Draw everything
            self.draw()
            
        # Done! Clean up
        pygame.quit()
        
    def process_events(self):
        """Process keyboard events"""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_g:  # Toggle grid
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_s:  # Toggle sensor range
                    self.show_sensor_range = not self.show_sensor_range
                elif event.key == pygame.K_e:  # Toggle EKF estimates
                    self.show_ekf = not self.show_ekf
        
        # Set wheel velocities based on keys
        keys = pygame.key.get_pressed()
        v_l = 0
        v_r = 0
        
        if keys[K_UP]:  # Forward
            v_l = 0.5
            v_r = 0.5
        if keys[K_DOWN]:  # Backward
            v_l = -0.5
            v_r = -0.5
        if keys[K_LEFT]:  # Turn left
            v_l -= 0.3
            v_r += 0.3
        if keys[K_RIGHT]:  # Turn right
            v_l += 0.3
            v_r -= 0.3
        
        self.robot.set_wheel_velocities(v_l, v_r)
        
    def process_observations(self):
        """Process observations of landmarks"""
        self.last_observations = []
        
        for i, (lx, ly) in enumerate(self.landmarks):
            r, phi = self.robot.measure_landmark(lx, ly)
            
            # If landmark is observed (within sensor range)
            if r is not None:
                # Add to last observations for display
                self.last_observations.append((i, r, phi))
                
                # Update EKF with this observation
                self.ekf.update(i, (r, phi))
        
    def draw(self):
        """Draw the world"""
        # Fill the screen with white
        self.screen.fill(self.WHITE)
        
        # Draw grid if enabled
        if self.show_grid:
            self.draw_grid()
        
        # Draw sensor range if enabled
        if self.show_sensor_range:
            pass
            self.draw_sensor_range()
        
        # Draw robot path
        self.draw_robot_path()
        
        # Draw landmarks
        self.draw_landmarks()
        
        # Draw EKF estimates if enabled
        if self.show_ekf:
            self.draw_ekf_estimates()
        
        # Draw robot
        self.draw_robot()
        
        # Draw observed landmarks
        self.draw_observations()
        
        # Draw debug info
        self.draw_debug_info()
        
        # Update display
        pygame.display.flip()
        
    def draw_grid(self):
        """Draw a grid for reference"""
        # Grid spacing in meters
        grid_spacing = 1.0
        
        # Convert to screen coordinates
        grid_pixels = int(grid_spacing * self.scale)
        
        # Draw vertical lines
        for x in range(0, self.width, grid_pixels):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.height))
        
        # Draw horizontal lines
        for y in range(0, self.height, grid_pixels):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.width, y))
            
    def draw_sensor_range(self):
        """Draw sensor range circle"""
        x, y = self.world_to_screen(self.robot.x, self.robot.y)
        radius = int(self.robot.sensor_range * self.scale)
        pygame.draw.circle(self.screen, (230, 230, 250), (x, y), radius, 1)
        
    def draw_robot_path(self):
        """Draw the robot's path"""
        path_points = [(self.world_to_screen(x, y)) for x, y in zip(self.robot.history['x'], self.robot.history['y'])]
        if len(path_points) > 1:
            pygame.draw.lines(self.screen, (200, 200, 255), False, path_points, 2)
    
    def draw_robot(self):
        """Draw the robot on the screen"""
        # Get robot position in screen coordinates
        x, y = self.world_to_screen(self.robot.x, self.robot.y)
        
        # Draw robot body
        robot_radius = int(0.15 * self.scale)  # 15cm radius
        pygame.draw.circle(self.screen, self.RED, (x, y), robot_radius)
        
        # Draw direction indicator (heading)
        end_x = x + int(robot_radius * np.cos(self.robot.theta))
        end_y = y - int(robot_radius * np.sin(self.robot.theta))  # Flip y-axis
        pygame.draw.line(self.screen, self.BLACK, (x, y), (end_x, end_y), 2)
        
    def draw_landmarks(self):
        """Draw landmarks"""
        for i, (lx, ly) in enumerate(self.landmarks):
            x, y = self.world_to_screen(lx, ly)
            # Draw landmark
            pygame.draw.circle(self.screen, self.BLUE, (x, y), 5)
            # Draw landmark ID
            text = self.font.render(str(i), True, self.BLACK)
            self.screen.blit(text, (x + 10, y - 10))
    
    def draw_ekf_estimates(self):
        """Draw EKF estimated landmarks"""
        # Draw estimated robot position
        if len(self.ekf.mu) >= 3:
            est_x, est_y = self.world_to_screen(self.ekf.mu[0, 0], self.ekf.mu[1, 0])
            pygame.draw.circle(self.screen, self.GREEN, (est_x, est_y), 8, 2)
        
        # Draw estimated landmarks
        for i, (lx, ly, uncertainty) in enumerate(self.ekf.estimated_landmarks):
            x, y = self.world_to_screen(lx, ly)
            
            # Scale ellipse size with uncertainty
            size = int(np.clip(uncertainty * 10, 10, 50))
            
            # Draw estimated landmark position
            pygame.draw.circle(self.screen, self.GREEN, (x, y), 5)
            
            # Draw uncertainty ellipse
            pygame.draw.ellipse(self.screen, self.GREEN, (x - size//2, y - size//2, size, size), 1)
            
            # Draw landmark ID
            text = self.font.render(f"E{i}", True, self.GREEN)
            self.screen.blit(text, (x + 15, y - 15))

    def draw_observations(self):
        """Draw lines to observed landmarks"""
        robot_x, robot_y = self.world_to_screen(self.robot.x, self.robot.y)
        
        for landmark_id, r, phi in self.last_observations:
            # Calculate endpoint of observation ray
            endpoint_x = self.robot.x + r * np.cos(phi + self.robot.theta)
            endpoint_y = self.robot.y + r * np.sin(phi + self.robot.theta)
            
            endpoint_screen_x, endpoint_screen_y = self.world_to_screen(endpoint_x, endpoint_y)
            
            # Draw ray
            pygame.draw.line(self.screen, self.ORANGE, (robot_x, robot_y), 
                            (endpoint_screen_x, endpoint_screen_y), 1)
            
            # Draw small circle at endpoint
            pygame.draw.circle(self.screen, self.ORANGE, (endpoint_screen_x, endpoint_screen_y), 3)
    
    def draw_debug_info(self):
        """Draw debug information"""
        # Display controls
        info_text = [
            "Controls: Arrow keys to move",
            "G: Toggle grid",
            "S: Toggle sensor range",
            "E: Toggle EKF estimates",
            f"Robot: ({self.robot.x:.2f}, {self.robot.y:.2f}, {self.robot.theta:.2f})",
            f"Landmarks: {len(self.landmarks)}",
            f"Observed: {len(self.last_observations)}"
        ]
        
        for i, text in enumerate(info_text):
            surface = self.font.render(text, True, self.BLACK)
            self.screen.blit(surface, (10, 10 + i * 25))
            
    def add_landmark(self, x, y):
        """Add a landmark to the environment"""
        self.landmarks.append((x, y))

def main():
    # Create simulator
    sim = Simulator(width=800, height=600)
    
    # Add some random landmarks
    np.random.seed(42)  # For reproducibility
    
    # Add landmarks in a grid pattern
    for x in range(1, 8, 2):
        for y in range(1, 6, 2):
            # Add some randomness to positions
            lx = x + np.random.uniform(-0.3, 0.3)
            ly = y + np.random.uniform(-0.3, 0.3)
            sim.add_landmark(lx, ly)
    
    # Run the simulator
    sim.run()
    
if __name__ == "__main__":
    main()