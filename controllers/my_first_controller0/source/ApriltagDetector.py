import cv2
import numpy as np
import dt_apriltags
import apriltag

class AprilTagDetector:
	"""A class for detecting and processing AprilTags in images."""

	def __init__(self, camera, robot):
		"""
		Initialize the AprilTag detector with camera and robot parameters.

		Args:
			camera: Camera object providing image capture capabilities
			robot: Robot object for navigation and control
		"""
		# Camera properties
		self.camera = camera
		self.width = camera.getWidth()
		self.height = camera.getHeight()
		self.foc_distance = camera.getFocalDistance()
		self.focal_length = camera.getFocalLength()
		self.robot = robot

		# Configure AprilTag detector with optimal parameters
		self.detector = dt_apriltags.Detector(
			families='tag36h11',
			nthreads=4,
			quad_decimate=1.0,  # Reduced from default to process full resolution
			quad_sigma=0.8,     # Increased Gaussian blur for better edge detection
			refine_edges=True,  # Enable edge refinement
			decode_sharpening=0.25,  # Add sharpening to improve decoding
			debug=False
		)
		
		# Storage for detected AprilTags
		self.april_tags = {}

	def detect(self, image) -> dict:
		"""
		Detect AprilTags in the given image and estimate their poses.

		Args:
			image: Input image to process

		Returns:
			dict: Dictionary mapping tag IDs to their pose matrices
		"""
		# Convert camera image to numpy array
		image = self.camera.getImage()
		image_array = np.frombuffer(image, np.uint8)
		rgb_image = image_array.reshape((self.height, self.width, 4))
		gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2GRAY)

		# Define camera calibration parameters
		self.focal_length = 205  # Calculated based on FOV and image size
		fx, fy = 2180.821770876292, 2022.5541555343623
		cx, cy = 971.9235543445766, 537.3153526641415
		camera_matrix = np.array([[fx, 0, cx],
								  [0, fy, cy],
								  [0, 0, 1]], dtype=np.float32)
		
		camera_matrix = [fx, fy, cx, cy]
		tags = self.detector.detect(gray, estimate_tag_pose=True, camera_params=camera_matrix, tag_size=0.128) # 0.48
		
		# sort the tags based on location
		# Store detected tag locations
		tag_locations = {}
		for tag in tags:
			tag_id = tag.tag_id
			# rotate the translation vector
			tvec = tag.pose_t
			tvec = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) @ tvec
			if tag_id not in tag_locations:
				tag_locations[tag_id] = [tvec]
			else:
				tag_locations[tag_id].append(tvec)
		
		for tag_id, tvec_list in tag_locations.items():
			tag_locations[tag_id] = sorted(
        	    tvec_list,
    	        key=lambda t: np.linalg.norm(t)
        	)
		print(tag_locations)
		return tag_locations
