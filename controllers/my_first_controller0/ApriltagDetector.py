import cv2
import numpy as np
import dt_apriltags

class AprilTagDetector:
	def __init__(self, camera, robot):
		# Initialize camera properties
		self.camera = camera
		self.width = camera.getWidth()
		self.height = camera.getHeight()
		self.foc_distance = camera.getFocalDistance()
		self.focal_length = camera.getFocalLength()
		self.robot = robot

		# Initialize AprilTag detector
		self.detector = dt_apriltags.Detector(
			families='tag36h11',
			nthreads = 4,
			quad_decimate=1,
			refine_edges=1,
			quad_sigma=0.0,
		)
		
		# Dictionary to hold ID and distance from the robot
		self.AprilTags = {}

	def detect(self, image):
		# Convert image buffer to numpy array

		image = self.camera.getImage()
		buf_as_np_array = np.frombuffer(image, np.uint8)
		rgb = buf_as_np_array.reshape((self.height, self.width, 4))
		gray = cv2.cvtColor(rgb, cv2.COLOR_BGRA2GRAY)

		# Set camera parameters
		self.focal_length = 205  # NOTE -This is a function of FOV and image size
		fx, fy = self.focal_length, self.focal_length
		cx, cy = self.width / 2, self.height / 2
		camera_params = [fx, fy, cx, cy]
		tag_locs = {}  # Key is tag ID, value is an array of pose matrices
		result = self.detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=0.6)
		
		# Process multiple detected tags
		if len(result) > 0:
			for r in result:
				r.pose_t = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) @ r.pose_t
				if r.tag_id not in tag_locs:
					tag_locs[r.tag_id] = [r.pose_t.flatten()]
				else:
					tag_locs[r.tag_id].append(r.pose_t.flatten())
		else:
			return
		# average the tags based on their id
		for tag_id, poses in tag_locs.items():
			print(f"Tag ID: {tag_id}, Poses: {poses}")
		print("_" * 20)
		#TODO - Use the Likelihood Ratio Test to determine if two tags are the same
		# print the tags
		return self.AprilTags
