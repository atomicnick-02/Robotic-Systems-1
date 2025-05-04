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
		gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2GRAY)

		# Define camera calibration parameters
		self.focal_length = 205  # Calculated based on FOV and image size
		fx, fy = 874.5497056724861, 876.8411106683374
		cx, cy = 1279.315, 722.6236
		
		camera_matrix = np.array([[fx, 0, cx],
								  [0, fy, cy],
								  [0, 0, 1]], dtype=np.float32)
		
		distortion = np.array([-0.4856, -20.664, -0.0049, -0.0007, 276.5148])
		# Store detected tag locations
		tag_locations = {}

		# Detect AprilTags in the image
		options = apriltag.DetectorOptions(families='tag36h11')
		detector = apriltag.Detector(options)
		results = detector.detect(gray)
		
		print(f"Detected {len(results)} tags")
		for result in results:
			corners = result.corners
			corners = np.array([[p[0], p[1]] for p in corners])
			tag_id = result.tag_id
			
			# Calculate distance
			distance, rvec, tvec = self.calculate_distance(corners, 0.6, camera_matrix, distortion)
			
			
			print(f"Tag ID: {tag_id}, Distance: {distance:.3f} meters")
			print(f"Translation Vector: {tvec.flatten()}")
			
		# cv2.imshow("AprilTag Detection", gray)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# Debug output for detected tags
		# for tag_id, poses in tag_locations.items():
		# 	print(f"Tag ID: {tag_id}, Poses: {poses}")
		# print("_" * 20)

		return tag_locations

	def calculate_distance(self,corners, tag_size, camera_matrix, dist_coeffs):
		"""
		Calculate distance to the tag from the camera
		
		Args:
			corners: The four corners of the detected tag
			tag_size: The physical size of the tag in meters
			camera_matrix: Camera intrinsic parameters
			dist_coeffs: Camera distortion coefficients
		
		Returns:
			distance: Distance from camera to tag in meters
			rvec: Rotation vector
			tvec: Translation vector
		"""
		# Define 3D points of the tag in its own coordinate system
		# The tag is centered at the origin with the specified size
		half_size = tag_size / 2
		obj_points = np.array([
			[-half_size, -half_size, 0],
			[ half_size, -half_size, 0],
			[ half_size,  half_size, 0],
			[-half_size,  half_size, 0]
		], dtype=np.float32)
		
		# Convert corners to the format expected by solvePnP
		img_points = np.array(corners, dtype=np.float32)
		
		# Solve for pose
		success, rvec, tvec = cv2.solvePnP(
			obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
		)
		
		# Calculate distance from camera to the center of the tag
		distance = np.linalg.norm(tvec)
		
		return distance, rvec, tvec
