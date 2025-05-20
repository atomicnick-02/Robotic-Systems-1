import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pygame
import math
from scipy.stats import chi2
import time
from matplotlib.transforms import Affine2D

# EKF SLAM with Unknown Correspondences
class EKFSLAM:
    def __init__(self, robot_init_state, landmark_size, motion_noise, measurement_noise):
        # Robot state: [x, y, theta]
        # Initialize state covariance matrix
        self.robot_state = robot_init_state
        self.landmark_size = landmark_size
        
        # Initialize the state vector: robot pose + all landmarks
        # Initially no landmarks are observed, so only robot state
        self.state = np.copy(robot_init_state)
        
        # Initialize covariance matrix (3x3 for the robot state only initially)
        self.covariance = np.diag([0.01, 0.01, 0.01])
        
        # Noise parameters
        self.motion_noise = motion_noise  # [noise_x, noise_y, noise_theta]
        self.measurement_noise = measurement_noise  # [noise_range, noise_bearing]
        
        # Landmark tracking
        self.landmarks = {}  # Dictionary to store landmark positions
        self.landmark_indices = {}  # Map landmark IDs to state vector indices
        
        # Mahalanobis distance threshold for data association (95% confidence)
        self.mahalanobis_threshold = chi2.ppf(0.95, 2)
        
    def predict(self, control):
        """
        EKF prediction step using differential drive motion model
        control: [v, w] - linear and angular velocity
        """
        v, w = control
        dt = 0.1  # time step
        
        # Extract current robot state
        x, y, theta = self.robot_state
        
        # Motion model (non-linear)
        if abs(w) < 1e-5:  # Handle the case when w is close to zero
            x_new = x + v * dt * np.cos(theta)
            y_new = y + v * dt * np.sin(theta)
            theta_new = theta
        else:
            x_new = x + (v/w) * (np.sin(theta + w*dt) - np.sin(theta))
            y_new = y + (v/w) * (np.cos(theta) - np.cos(theta + w*dt))
            theta_new = theta + w * dt
            
        # Update robot state
        self.robot_state = np.array([x_new, y_new, theta_new])
        
        # Jacobian of motion model with respect to robot state
        if abs(w) < 1e-5:
            G = np.array([
                [1, 0, -v * dt * np.sin(theta)],
                [0, 1, v * dt * np.cos(theta)],
                [0, 0, 1]
            ])
        else:
            G = np.array([
                [1, 0, (v/w) * (np.cos(theta + w*dt) - np.cos(theta))],
                [0, 1, (v/w) * (np.sin(theta + w*dt) - np.sin(theta))],
                [0, 0, 1]
            ])
        
        # Motion noise
        R = np.diag([
            self.motion_noise[0]**2,
            self.motion_noise[1]**2,
            self.motion_noise[2]**2
        ])
        
        # Update state vector (robot part only)
        self.state[:3] = self.robot_state
        
        # Expand G to full state size (for both robot and landmarks)
        G_full = np.eye(len(self.state))
        G_full[:3, :3] = G
        
        # Update full covariance
        R_full = np.zeros((len(self.state), len(self.state)))
        R_full[:3, :3] = R
        self.covariance = G_full @ self.covariance @ G_full.T + R_full
        
        return self.robot_state
    
    def update(self, measurements, detected_landmarks):
        """
        EKF update step with measurement model
        measurements: list of [range, bearing] to different landmarks
        detected_landmarks: list of true landmark IDs (for simulation)
        """
        for i, (r, b) in enumerate(measurements):
            true_id = detected_landmarks[i]
            
            # Convert measurement to global coordinates for visualization
            lm_x = self.robot_state[0] + r * np.cos(b + self.robot_state[2])
            lm_y = self.robot_state[1] + r * np.sin(b + self.robot_state[2])
            
            # Check if this is a new landmark
            if true_id not in self.landmarks:
                # Add new landmark to state vector
                self.add_landmark(true_id, [lm_x, lm_y])
            
            # Now perform the EKF update for this measurement
            self.update_landmark(true_id, [r, b])
    
    def add_landmark(self, landmark_id, landmark_pos):
        """
        Add a new landmark to the state vector and covariance matrix
        """
        # Current size of state vector
        n = len(self.state)
        
        # Store landmark position for visualization
        self.landmarks[landmark_id] = landmark_pos
        
        # Map landmark ID to its position in state vector
        self.landmark_indices[landmark_id] = n
        
        # Extend state vector with new landmark
        self.state = np.append(self.state, landmark_pos)
        
        # Extend covariance matrix
        new_cov = np.zeros((n+2, n+2))
        new_cov[:n, :n] = self.covariance
        new_cov[n:, n:] = np.diag([1000, 1000])  # High initial uncertainty for new landmark
        
        # Update the full covariance
        self.covariance = new_cov
    
    def update_landmark(self, landmark_id, measurement):
        """
        Update state for a measured landmark
        """
        # Get landmark index in state vector
        idx = self.landmark_indices[landmark_id]
        
        # Extract landmark position from state
        lm_x, lm_y = self.state[idx:idx+2]
        
        # Extract robot pose
        x, y, theta = self.state[:3]
        
        # Calculate expected measurement
        dx = lm_x - x
        dy = lm_y - y
        q = dx**2 + dy**2
        q_sqrt = np.sqrt(q)
        
        # Expected measurement [range, bearing]
        z_expected = np.array([
            q_sqrt,
            np.arctan2(dy, dx) - theta
        ])
        
        # Wrap bearing to [-pi, pi]
        z_expected[1] = self.wrap_to_pi(z_expected[1])
        
        # Compute innovation (measurement residual)
        innovation = np.array(measurement) - z_expected
        innovation[1] = self.wrap_to_pi(innovation[1])
        
        # Jacobian of measurement model w.r.t. state
        H = np.zeros((2, len(self.state)))
        
        # Derivatives w.r.t. robot pose
        H[0, 0] = -dx / q_sqrt
        H[0, 1] = -dy / q_sqrt
        H[0, 2] = 0
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1
        
        # Derivatives w.r.t. landmark position
        H[0, idx] = dx / q_sqrt
        H[0, idx+1] = dy / q_sqrt
        H[1, idx] = -dy / q
        H[1, idx+1] = dx / q
        
        # Measurement noise
        R = np.diag([
            self.measurement_noise[0]**2,
            self.measurement_noise[1]**2
        ])
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + R
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ innovation
        
        # Update covariance
        self.covariance = (np.eye(len(self.state)) - K @ H) @ self.covariance
        
        # Update robot state
        self.robot_state = self.state[:3]
        
        # Update landmark position in dictionary (for visualization)
        self.landmarks[landmark_id] = self.state[idx:idx+2]
    
    def wrap_to_pi(self, angle):
        """Wrap angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def data_association(self, measurements):
        """
        Associate measurements with landmarks (unknown correspondences)
        Returns list of landmark IDs for each measurement
        """
        associated_landmarks = []
        
        for z in measurements:
            best_landmark = None
            min_distance = float('inf')
            
            # For each existing landmark, calculate Mahalanobis distance
            for landmark_id, landmark_idx in self.landmark_indices.items():
                # Expected measurement for this landmark
                lm_x, lm_y = self.state[landmark_idx:landmark_idx+2]
                x, y, theta = self.state[:3]
                
                dx = lm_x - x
                dy = lm_y - y
                q = dx**2 + dy**2
                q_sqrt = np.sqrt(q)
                
                z_expected = np.array([
                    q_sqrt,
                    np.arctan2(dy, dx) - theta
                ])
                z_expected[1] = self.wrap_to_pi(z_expected[1])
                
                # Innovation
                innovation = np.array(z) - z_expected
                innovation[1] = self.wrap_to_pi(innovation[1])
                
                # Jacobian
                H = np.zeros((2, len(self.state)))
                
                # Derivatives w.r.t. robot pose
                H[0, 0] = -dx / q_sqrt
                H[0, 1] = -dy / q_sqrt
                H[0, 2] = 0
                H[1, 0] = dy / q
                H[1, 1] = -dx / q
                H[1, 2] = -1
                
                # Derivatives w.r.t. landmark position
                H[0, landmark_idx] = dx / q_sqrt
                H[0, landmark_idx+1] = dy / q_sqrt
                H[1, landmark_idx] = -dy / q
                H[1, landmark_idx+1] = dx / q
                
                # Measurement noise
                R = np.diag([
                    self.measurement_noise[0]**2,
                    self.measurement_noise[1]**2
                ])
                
                # Innovation covariance
                S = H @ self.covariance @ H.T + R
                
                # Mahalanobis distance
                distance = innovation.T @ np.linalg.inv(S) @ innovation
                
                if distance < min_distance:
                    min_distance = distance
                    best_landmark = landmark_id
            
            # Check if the best match is below threshold
            if min_distance < self.mahalanobis_threshold:
                associated_landmarks.append(best_landmark)
            else:
                # Create new landmark
                new_id = len(self.landmarks) if not self.landmarks else max(self.landmarks.keys()) + 1
                associated_landmarks.append(new_id)
                
                # Convert measurement to global coordinates for new landmark
                r, b = z
                x, y, theta = self.state[:3]
                lm_x = x + r * np.cos(b + theta)
                lm_y = y + r * np.sin(b + theta)
                
                # Don't add the landmark yet, will be added during update
                self.landmarks[new_id] = [lm_x, lm_y]
                
        return associated_landmarks


# Simulation Environment
class Environment:
    def __init__(self, width=10, height=10, n_landmarks=20):
        self.width = width
        self.height = height
        self.landmarks = {}
        
        # Create grid-like landmarks with some randomness
        grid_points_x = np.linspace(1, width-1, 5)
        grid_points_y = np.linspace(1, height-1, 5)
        
        landmark_id = 0
        for x in grid_points_x:
            for y in grid_points_y:
                # Add some randomness to grid points
                rand_x = x + np.random.uniform(-0.5, 0.5)
                rand_y = y + np.random.uniform(-0.5, 0.5)
                
                # Keep within bounds
                rand_x = max(0.5, min(width-0.5, rand_x))
                rand_y = max(0.5, min(height-0.5, rand_y))
                
                self.landmarks[landmark_id] = [rand_x, rand_y]
                landmark_id += 1
                
                # Stop if we have enough landmarks
                if landmark_id >= n_landmarks:
                    break
            if landmark_id >= n_landmarks:
                break
    
    def get_measurements(self, robot_pose, max_range=5.0, fov=120):
        """
        Get range and bearing measurements to landmarks within sensor range
        Returns: 
        - measurements: list of [range, bearing]
        - detected_landmarks: list of landmark IDs
        """
        x, y, theta = robot_pose  # Robot pose [x, y, theta]
        measurements = []
        detected_landmarks = []
        
        for landmark_id, landmark_pos in self.landmarks.items():
            # Calculate true range and bearing to landmark
            dx = landmark_pos[0] - x
            dy = landmark_pos[1] - y
            range_to_landmark = np.sqrt(dx**2 + dy**2)
            bearing_to_landmark = np.arctan2(dy, dx) - theta
            
            # Wrap bearing to [-pi, pi]
            bearing_to_landmark = (bearing_to_landmark + np.pi) % (2 * np.pi) - np.pi
            
            # Check if landmark is within sensor range and field of view
            fov_rad = np.radians(fov / 2)
            if range_to_landmark <= max_range and abs(bearing_to_landmark) <= fov_rad:
                # Add noise to measurements
                noisy_range = range_to_landmark + np.random.normal(0, 0.1)  # 10cm std dev
                noisy_bearing = bearing_to_landmark + np.random.normal(0, 0.05)  # 0.05 rad std dev
                
                measurements.append([noisy_range, noisy_bearing])
                detected_landmarks.append(landmark_id)
                
        return measurements, detected_landmarks


# Differential Drive Robot Simulator
class DifferentialDriveRobot:
    def __init__(self, init_state, wheel_radius=0.1, wheel_base=0.5):
        # State: [x, y, theta]
        self.state = np.array(init_state)
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        
        # Control inputs: [v, w] - linear velocity and angular velocity
        self.control = np.array([0.0, 0.0])
        
        # Physical constraints
        self.max_velocity = 10.0  # m/s
        self.max_angular_velocity = 1.5  # rad/s
        
    def update_state(self, dt):
        """
        Update robot state using differential drive kinematics
        """
        v, w = self.control
        
        # Apply motion model
        if abs(w) < 1e-6:  # Straight line motion
            self.state[0] += v * dt * np.cos(self.state[2])
            self.state[1] += v * dt * np.sin(self.state[2])
        else:  # Circular motion
            self.state[0] += (v / w) * (np.sin(self.state[2] + w * dt) - np.sin(self.state[2]))
            self.state[1] += (v / w) * (np.cos(self.state[2]) - np.cos(self.state[2] + w * dt))
            self.state[2] += w * dt
            
        # Normalize angle to [-pi, pi]
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        
        return self.state
    
    def set_control(self, v, w):
        """
        Set control inputs with constraints
        """
        self.control[0] = max(-self.max_velocity, min(self.max_velocity, v))
        self.control[1] = max(-self.max_angular_velocity, min(self.max_angular_velocity, w))


# Main Simulation Class
class EKFSLAMSimulation:
    def __init__(self):
        # Environment setup
        self.env_width = 20
        self.env_height = 20
        self.environment = Environment(width=self.env_width, height=self.env_height, n_landmarks=25)
        
        # Robot setup
        init_state = [self.env_width/2, self.env_height/2, 0]  # center of environment
        self.robot = DifferentialDriveRobot(init_state)
        
        # SLAM setup
        motion_noise = [0.1, 0.1, 0.05]  # [x, y, theta]
        measurement_noise = [0.2, 0.1]  # [range, bearing]
        self.slam = EKFSLAM(init_state, 2, motion_noise, measurement_noise)
        
        # Simulation parameters
        self.dt = 0.1  # simulation time step
        self.sensor_range = 5.0  # max sensing range
        self.fov = 120  # degrees
        
        # Visualization setup
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('EKF SLAM Simulation')
        
        # Initialize pygame for keyboard control
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption('Robot Control (Use arrow keys)')
        
        # Control variables
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.running = True
        
        # For animation
        self.robot_marker = None
        self.landmark_markers = {}
        self.estimated_landmark_markers = {}
        self.ellipse_markers = {}
        self.robot_path = []
        self.estimated_path = []
        
        # Robot triangle coordinates
        self.robot_size = 0.5
        self.robot_triangle = None
        self.estimated_robot_triangle = None
        
    def update_animation(self):
        """Update the animation elements"""
        self.ax.clear()
        
        # Set plot limits
        self.ax.set_xlim(0, self.env_width)
        self.ax.set_ylim(0, self.env_height)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title('EKF SLAM Simulation')
        
        # Plot true landmarks
        for landmark_id, pos in self.environment.landmarks.items():
            self.ax.plot(pos[0], pos[1], 'bs', markersize=8, label='True Landmarks' if landmark_id == 0 else "")
        
        # Plot estimated landmarks with uncertainty ellipses
        for landmark_id, pos in self.slam.landmarks.items():
            self.ax.plot(pos[0], pos[1], 'ro', markersize=6, label='Estimated Landmarks' if landmark_id == 0 else "")
            
            # Get landmark index in state vector
            if landmark_id in self.slam.landmark_indices:
                idx = self.slam.landmark_indices[landmark_id]
                
                # Extract covariance submatrix for this landmark
                landmark_cov = self.slam.covariance[idx:idx+2, idx:idx+2]
                
                # Plot uncertainty ellipse (95% confidence)
                eigenvalues, eigenvectors = np.linalg.eig(landmark_cov)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                
                # Scale for 95% confidence
                scale = 2.447  # chi-square with 2 DOF, 95% confidence
                width = 2 * scale * np.sqrt(eigenvalues[0])
                height = 2 * scale * np.sqrt(eigenvalues[1])
                
                ellipse = patches.Ellipse(
                    (pos[0], pos[1]),
                    width, height,
                    angle=angle,
                    fill=False,
                    color='r',
                    alpha=0.5
                )
                self.ax.add_patch(ellipse)
        
        # Plot robot
        x, y, theta = self.robot.state
        
        # Robot as a triangle
        triangle_points = np.array([
            [self.robot_size, 0],
            [-self.robot_size/2, self.robot_size/2],
            [-self.robot_size/2, -self.robot_size/2]
        ])
        
        # Rotate and translate triangle
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotated_points = np.dot(triangle_points, rotation.T)
        robot_points = rotated_points + np.array([x, y])
        
        # Plot true robot
        true_robot = plt.Polygon(robot_points, fill=True, color='blue', alpha=0.7, label='True Robot')
        self.ax.add_patch(true_robot)
        
        # Plot estimated robot
        x_est, y_est, theta_est = self.slam.robot_state
        rotation_est = np.array([
            [np.cos(theta_est), -np.sin(theta_est)],
            [np.sin(theta_est), np.cos(theta_est)]
        ])
        rotated_points_est = np.dot(triangle_points, rotation_est.T)
        robot_points_est = rotated_points_est + np.array([x_est, y_est])
        
        est_robot = plt.Polygon(robot_points_est, fill=True, color='red', alpha=0.7, label='Estimated Robot')
        self.ax.add_patch(est_robot)
        
        # Plot robot direction line (heading)
        heading_length = self.robot_size
        self.ax.plot([x, x + heading_length * np.cos(theta)], 
                    [y, y + heading_length * np.sin(theta)], 
                    'b-', linewidth=2)
        
        # Plot estimated heading
        self.ax.plot([x_est, x_est + heading_length * np.cos(theta_est)], 
                    [y_est, y_est + heading_length * np.sin(theta_est)], 
                    'r-', linewidth=2)
        
        # Plot paths
        if self.robot_path:
            path_array = np.array(self.robot_path)
            self.ax.plot(path_array[:, 0], path_array[:, 1], 'b-', alpha=0.5, label='True Path')
        
        if self.estimated_path:
            est_path_array = np.array(self.estimated_path)
            self.ax.plot(est_path_array[:, 0], est_path_array[:, 1], 'r-', alpha=0.5, label='Estimated Path')
        
        # Plot sensor field of view
        sensor_arc = np.linspace(-np.radians(self.fov/2), np.radians(self.fov/2), 50)
        sensor_x = x + self.sensor_range * np.cos(sensor_arc + theta)
        sensor_y = y + self.sensor_range * np.sin(sensor_arc + theta)
        self.ax.plot([x] * len(sensor_arc), [y] * len(sensor_arc), 'g-', alpha=0.2)
        self.ax.plot([x, x], [y, y], 'g-', alpha=0.2)
        self.ax.plot(sensor_x, sensor_y, 'g-', alpha=0.2)
        
        # Plot legend
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Status text
        status_text = f"Linear Velocity: {self.linear_velocity:.2f} m/s\n" \
                      f"Angular Velocity: {self.angular_velocity:.2f} rad/s\n" \
                      f"Landmarks: {len(self.slam.landmarks)}/{len(self.environment.landmarks)}"
        self.ax.text(0.02, 0.98, status_text, transform=self.ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
    def handle_events(self):
        """Handle pygame events for user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                plt.close()
                return
        
        # Get pressed keys
        keys = pygame.key.get_pressed()
        
        # Reset velocities
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        
        # Update velocities based on key presses
        if keys[pygame.K_UP]:
            self.linear_velocity = 4.0
        if keys[pygame.K_DOWN]:
            self.linear_velocity = -4.0
        if keys[pygame.K_LEFT]:
            self.angular_velocity = 1.0
        if keys[pygame.K_RIGHT]:
            self.angular_velocity = -1.0
            
        # Set robot control
        self.robot.set_control(self.linear_velocity, self.angular_velocity)
    
    def run_simulation(self):
        """Main simulation loop"""
        plt.ion()  # Turn on interactive mode
        
        last_time = time.time()
        
        try:
            while self.running:
                # Handle user input
                self.handle_events()
                
                # Control timing
                current_time = time.time()
                if current_time - last_time >= self.dt:
                    # Update robot state
                    self.robot.update_state(self.dt)
                    
                    # Update SLAM prediction with control input
                    self.slam.predict(self.robot.control)
                    
                    # Get sensor measurements
                    measurements, true_landmarks = self.environment.get_measurements(
                        self.robot.state, self.sensor_range, self.fov)
                    
                    # Only update if we have measurements
                    if measurements:
                        # In a real system, we wouldn't know the true landmark IDs,
                        # so we use data association instead
                        # detected_landmarks = self.slam.data_association(measurements)
                        
                        # For educational purposes, we'll use the true associations
                        self.slam.update(measurements, true_landmarks)
                    
                    # Store path data
                    self.robot_path.append(self.robot.state[:2].copy())
                    self.estimated_path.append(self.slam.robot_state[:2].copy())
                    
                    # Update visualization
                    self.update_animation()
                    plt.draw()
                    plt.pause(0.001)
                    
                    last_time = current_time
                
                # Keep pygame window responsive
                pygame.display.flip()
        
        except KeyboardInterrupt:
            print("Simulation terminated by user.")
        finally:
            pygame.quit()
            plt.close()


# Entry point
if __name__ == "__main__":
    simulation = EKFSLAMSimulation()
    simulation.run_simulation()