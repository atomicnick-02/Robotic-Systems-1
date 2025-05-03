import cv2
import numpy as np
import dt_apriltags


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
			quad_decimate=1,
			refine_edges=1,
			quad_sigma=0.0,
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
		gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2GRAY)

		# Define camera calibration parameters
		# self.focal_length = 205  # Calculated based on FOV and image size
		print("FOV:", self.camera.getFov())
		fx, fy = self.focal_length, self.focal_length
		cx, cy = 1280 / 2, 900 / 2
		camera_params = [fx, fy, cx, cy] 
		# print(f"focal length: {self.focal_length} pixel size: {pixel_size}")
		print("camera_params:", camera_params)
		# Store detected tag locations
		tag_locations = {}

		# Detect tags and estimate their poses
		detections = self.detector.detect(
			gray_image,
			estimate_tag_pose=True,
			camera_params=camera_params,
			tag_size=0.3
		)
		
		# Process detected tags
		if not detections:
			return None

		# Transform and store pose information for each detected tag
		for detection in detections:
			# Apply coordinate transformation
			detection.pose_t = np.array([
				[0, 0, 1],
				[1, 0, 0],
				[0, 1, 0]
			]) @ detection.pose_t

			if detection.tag_id not in tag_locations:
				tag_locations[detection.tag_id] = [detection.pose_t.flatten()]
			else:
				tag_locations[detection.tag_id].append(detection.pose_t.flatten())

		# Debug output for detected tags
		# for tag_id, poses in tag_locations.items():
		# 	print(f"Tag ID: {tag_id}, Poses: {poses}")
		# print("_" * 20)

		return tag_locations
