from controllers.test_files.test_slam.EKFSlam_KL import EKF_SLAM
import matplotlib.pyplot as plt
import numpy as np


def angle_dist(b, a):
	theta = b - a
	while theta < -np.pi:
		theta += 2. * np.pi
	while theta > np.pi:
		theta -= 2. * np.pi
	return theta

def robot_landmark_2d_measurement(robot_state, landmark_state):
	# robot_state: x = [θ, x, y]^T (numpy 3x1 array)
	# landmark_state = (m_x, m_y) (tuple0
	### TO-DO: Compute the 2D measurement for the robot_state and the landmark_state
	### You should return a tuple with (distance, relative_angle)
	### ANSWER: Insert code here
	mx = landmark_state[0]
	my = landmark_state[1]
	theta = robot_state[0,0]
	rx = robot_state[1,0]
	ry = robot_state[2,0]
	dist = np.sqrt(( mx - rx)**2 +(my-ry)**2)

	phi = np.atan2(my - ry, mx - rx) - theta
	phi = angle_dist(phi, 0) 

	return (dist, phi)

def robot_move(u, state, dt):

	v, w = u
	x, y, theta = state
	# Update state using odometry model
	x += v * np.cos(theta) * dt
	y += v * np.sin(theta) * dt
	theta += w * dt
	R = np.diag([0.1, 0.1, np.deg2rad(0.1)]) ** 2
	Q = np.diag([1, np.deg2rad(5)]) ** 2
	# Add noise to the state
	x += np.random.normal(0, np.sqrt(R[0,0]))
	y += np.random.normal(0, np.sqrt(R[1,1]))
	theta += np.random.normal(0, np.sqrt(R[2,2]))

	return [x, y, theta]

def lidar(x, landmarks, width = 2. * np.pi, noise = 1e-3):
    detects = []

    ### TO-DO: Compute all possible detects. You should fill the list detects with tuples of the form (distance, relative_angle, landmark id)
    ### The landmark ids should be 1-based.
    ### You should add Gaussian noise to the distance and relative angle with zero mean and `noise` as variance
    for k in range(len(landmarks)):
        ### ANSWER: Insert code here
        dist, phi = robot_landmark_2d_measurement(x,landmarks[k])
        if phi < width/2 and phi > - width / 2:
            # i have detected a landmark
            dist += np.random.normal(0,noise)
            phi += np.random.normal(0,noise)
            detects.append((dist,phi,k+1))
        ### END of ANSWER
    return detects

def main():
	# Initialize the robot
	slam = EKF_SLAM()
	
	# Set parameters if needed
	slam.set_time_step(0.2)  # 100ms time step
	time_steps = 100  # Number of time steps to simulate
	# Example control and measurement
	u = [0, np.deg2rad(10)]  # Move forward at 1m/s with 5deg/s rotation
	landmarks = [(4., 4.), (4., 0.), (4., -4.), (0., -4.), (-4., -4.), (-4., 0.), (-4., 4.), (0., 4.)]
	
	robot_pose, landmarks = slam.update(u, z)
	detected_landmarks = []
	real_trajectory = np.zeros((time_steps, 3))
	estimated_trajectory = np.zeros((time_steps, 3))
	fig, ax = plt.subplots()
	
	for t in range(time_steps):
		# Simulate control and measurement
		u = [0.0, np.deg2rad(8)]
		# Update SLAM with control and measurement

		real_trajectory[t] = robot_move(u, robot_pose, 0.2)
		landmark_readings =  [] 


		estimated_trajectory[t] = robot_pose
		robot_pose, landmarks = slam.update(u, )
		
		# Print the robot pose and detected landmarks
		print(f"Robot pose: x={robot_pose[0]:.2f}, y={robot_pose[1]:.2f}, θ={np.rad2deg(robot_pose[2]):.2f}°")
		if landmarks is not None:
			print(f"Detected {len(landmarks)} landmarks:")
			for i, lm in enumerate(landmarks):
				print(f"  Landmark {i+1}: ({lm[0]:.2f}, {lm[1]:.2f})")
				detected_landmarks.append(lm)
		else:
			print("No landmarks detected.")
		# Update the plot
		ax.clear()
		ax.set_xlim(-10, 10)
		ax.set_ylim(-10, 10)
		ax.set_aspect('equal')
		ax.set_title("EKF SLAM Simulation")
		ax.set_xlabel("X position (m)")
		ax.set_ylabel("Y position (m)")
		# Plot the real trajectory
		real_trajectory = np.array(real_trajectory)
		estimated_trajectory = np.array(estimated_trajectory)
		ax.plot(real_trajectory[:, 0], real_trajectory[:, 1], label="Real Trajectory", color='blue')
		
		ax.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], label="Estimated Trajectory", color='red')
		# Plot the landmarks
		if landmarks is not None:
			landmarks = np.array(landmarks)
			ax.scatter(landmarks[:, 0], landmarks[:, 1], label="Landmarks", color='green')
		# Plot the robot position
		ax.scatter(robot_pose[0], robot_pose[1], label="Robot Position", color='orange')
		ax.legend()
		plt.pause(0.001)

	# Close the plot
	plt.show()

	# Get results
	print(f"Robot pose: x={robot_pose[0]:.2f}, y={robot_pose[1]:.2f}, θ={np.rad2deg(robot_pose[2]):.2f}°")
	if landmarks is not None:
		print(f"Detected {len(landmarks)} landmarks:")
		for i, lm in enumerate(landmarks):
			print(f"  Landmark {i+1}: ({lm[0]:.2f}, {lm[1]:.2f})")
		detected_landmarks.append(lm)
	else:
		print("No landmarks detected.")


if __name__ == "__main__":
	main()