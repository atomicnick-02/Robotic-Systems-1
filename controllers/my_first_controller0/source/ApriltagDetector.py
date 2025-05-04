import cv2
import numpy as np
import dt_apriltags
import glob
import os

class AprilTagCalibrator:
	"""A class for camera calibration using AprilGrid tags."""
	def __init__(self, tag_family='tag36h11', grid_size=80, tag_count=(6, 6), tag_size=10, tag_spacing=2.0):
		"""
		Initialize the AprilTag calibrator.
		
		Args:
			tag_family: AprilTag family (default: 'tag36h11')
			grid_size: Size of the grid in cm (default: 80x80)
			tag_count: Number of tags in the grid (rows, cols)
			tag_size: Size of each tag in cm
			tag_spacing: Spacing between tags relative to tag size
		"""
		# Configure AprilTag detector
		self.detector = dt_apriltags.Detector(
			families=tag_family,
			nthreads=4,
			quad_decimate=1.0,
			quad_sigma=0.0,
			refine_edges=1,
			decode_sharpening=0.25
		)
		
		# Calibration parameters
		self.tag_count = tag_count
		self.tag_size = tag_size  # cm
		self.tag_spacing = tag_spacing
		self.grid_size = grid_size  # cm
		
		# Storage for calibration data
		self.object_points = []  # 3D points in real world space
		self.image_points = []   # 2D points in image plane
		
		# Camera calibration results
		self.camera_matrix = None
		self.dist_coeffs = None
		self.rvecs = None
		self.tvecs = None
		
		# Create object points for the AprilGrid
		self._create_object_points()
	
	def _create_object_points(self):
		"""Create the 3D points corresponding to the AprilGrid pattern."""
		self.grid_points = {}
		
		# Calculate the actual tag size in the grid
		effective_tag_size = self.tag_size
		
		# Calculate the spacing between tags
		spacing = effective_tag_size * self.tag_spacing
		
		# Create points for each tag in the grid
		for row in range(self.tag_count[0]):
			for col in range(self.tag_count[1]):
				# Calculate the tag ID based on position in grid
				tag_id = row * self.tag_count[1] + col
				
				# Calculate the position of the four corners of this tag
				tag_corners = []
				x = col * (effective_tag_size + spacing)
				y = row * (effective_tag_size + spacing)
				
				# Add the four corners of the tag in consistent order
				# Bottom-left, bottom-right, top-right, top-left
				tag_corners.append([x, y, 0])
				tag_corners.append([x + effective_tag_size, y, 0])
				tag_corners.append([x + effective_tag_size, y + effective_tag_size, 0])
				tag_corners.append([x, y + effective_tag_size, 0])
				
				self.grid_points[tag_id] = np.array(tag_corners, dtype=np.float32)
	
	def process_image(self, image):
		"""
		Process an image to extract AprilTag corners for calibration.
		
		Args:
			image: Image to process (numpy array)
			
		Returns:
			bool: True if image was successfully processed
		"""
		# Convert to grayscale if needed
		if len(image.shape) == 3:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		else:
			gray = image
		
		# Detect AprilTags in the image
		detections = self.detector.detect(gray)
		
		if len(detections) < 4:  # Need minimum number of tags for reliable calibration
			print(f"Warning: Only {len(detections)} tags detected, skipping image")
			return False
		
		# Temporary storage for this image
		img_points = []
		obj_points = []
		
		# Process each detected tag
		for detection in detections:
			tag_id = detection.tag_id
			
			# Check if this tag is part of our grid
			if tag_id in self.grid_points:
				# Add image points (detected corners)
				for corner in detection.corners:
					img_points.append(corner)
				
				# Add corresponding object points
				for obj_corner in self.grid_points[tag_id]:
					obj_points.append(obj_corner)
		
		# Only add points if we detected sufficient corners
		if len(img_points) >= 4:
			self.image_points.append(np.array(img_points, dtype=np.float32))
			self.object_points.append(np.array(obj_points, dtype=np.float32))
			return True
		
		return False
	
	def calibrate(self, image_shape):
		"""
		Perform camera calibration based on collected points.
		
		Args:
			image_shape: Tuple (height, width) of the images
			
		Returns:
			dict: Calibration results
		"""
		if len(self.object_points) < 1:
			raise ValueError("Not enough images were successfully processed for calibration")
		
		# Perform camera calibration
		ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
			self.object_points, self.image_points, 
			(image_shape[1], image_shape[0]),  # width, height
			None, None
		)
		
		# Calculate reprojection error
		mean_error = 0
		for i in range(len(self.object_points)):
			imgpoints2, _ = cv2.projectPoints(
				self.object_points[i], self.rvecs[i], self.tvecs[i], 
				self.camera_matrix, self.dist_coeffs
			)
			error = cv2.norm(self.image_points[i], imgpoints2.reshape(-1, 2), cv2.NORM_L2) / len(imgpoints2)
			mean_error += error
		
		reprojection_error = mean_error / len(self.object_points) if self.object_points else float('inf')
		
		# Extract camera parameters
		fx = self.camera_matrix[0, 0]
		fy = self.camera_matrix[1, 1]
		cx = self.camera_matrix[0, 2]
		cy = self.camera_matrix[1, 2]
		
		return {
			'camera_matrix': self.camera_matrix,
			'dist_coeffs': self.dist_coeffs,
			'reprojection_error': reprojection_error,
			'fx': fx,
			'fy': fy,
			'cx': cx,
			'cy': cy
		}


class AprilTagDetector:
	"""A class for detecting and processing AprilTags in images."""
	def __init__(self, camera, robot, camera_params=None):
		"""
		Initialize the AprilTag detector with camera and robot parameters.
		
		Args:
			camera: Camera object providing image capture capabilities
			robot: Robot object for navigation and control
			camera_params: Optional camera parameters from calibration
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
		
		# Camera parameters (can be set manually or from calibration)
		self.camera_params = camera_params
	
	def calibrate_camera(self, calibration_images_path, tag_count=(6, 6), tag_size=10, tag_spacing=2.0):
		"""
		Calibrate camera using AprilGrid pattern.
		
		Args:
			calibration_images_path: Path to folder containing calibration images
			tag_count: Number of tags in the grid (rows, cols)
			tag_size: Size of each tag in cm
			tag_spacing: Spacing between tags relative to tag size
			
		Returns:
			dict: Calibration parameters
		"""
		# Create calibrator
		calibrator = AprilTagCalibrator(
			tag_family='tag36h11',
			grid_size=80,  # 80x80 cm grid
			tag_count=tag_count,
			tag_size=tag_size,
			tag_spacing=tag_spacing
		)
		
		# Load and process calibration images
		successful_images = 0
		image_files = glob.glob(os.path.join(calibration_images_path, '*.jpg')) + \
					 glob.glob(os.path.join(calibration_images_path, '*.png'))
		
		if not image_files:
			raise ValueError(f"No image files found in {calibration_images_path}")
		
		for img_file in image_files:
			print(f"Processing {img_file}")
			image = cv2.imread(img_file)
			if image is None:
				print(f"Failed to read {img_file}")
				continue
				
			if calibrator.process_image(image):
				successful_images += 1
				
		print(f"Successfully processed {successful_images} out of {len(image_files)} images")
		
		if successful_images < 3:
			raise ValueError("Too few images were successfully processed. Need at least 3.")
			
		# Perform calibration
		calib_results = calibrator.calibrate(image.shape[:2])
		
		# Store camera parameters
		self.camera_params = [
			calib_results['fx'], 
			calib_results['fy'], 
			calib_results['cx'], 
			calib_results['cy']
		]
		
		print(f"Camera calibrated successfully:")
		print(f"Camera matrix: \n{calib_results['camera_matrix']}")
		print(f"Distortion coefficients: {calib_results['dist_coeffs']}")
		print(f"Reprojection error: {calib_results['reprojection_error']}")
		
		return calib_results
	
	def detect(self, image=None) -> dict:
		"""
		Detect AprilTags in the given image and estimate their poses.
		
		Args:
			image: Input image to process (if None, capture from camera)
			
		Returns:
			dict: Dictionary mapping tag IDs to their pose matrices
		"""
		# Get camera image if not provided
		gray_image = None
		if image is None:
			image = self.camera.getImage()
			image_array = np.frombuffer(image, np.uint8)
			rgb_image = image_array.reshape((self.height, self.width, 4))
			gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2GRAY)
		elif isinstance(image, np.ndarray):
			# Convert to grayscale if needed
			if len(image.shape) == 3:
				gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			else:
				gray_image = image
		
		# Check if we have camera parameters
		if not self.camera_params:
			print("Warning: Camera not calibrated, using approximate parameters")
			# Use approximate parameters based on image size
			fx, fy = 290.650, 581.778
			cx, cy = 1294.43, 746.4
			camera_params = [fx, fy, cx, cy]
		else:
			camera_params = self.camera_params
			
		print("Using camera_params:", camera_params)
		
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
		
		# Save debug image if needed
		cv2.imwrite("detected_tags.png", gray_image)
		
		return tag_locations

 