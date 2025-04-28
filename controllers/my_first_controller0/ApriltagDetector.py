import cv2
import numpy as np
import apriltag
import wpilib.cameraserver
 
# supress scientific notation and use 4 decimal places
# np.set_printoptions(suppress=True, precision=4)
class AprilTagDetector:
	def __init__(self, camera, robot):
		self.camera = camera
		self.width = camera.getWidth()
		self.height = camera.getHeight()
		self.focal_length = camera.getFocalLength()
		self.robot = robot
		options = apriltag.DetectorOptions(
			families='tag36h11',
			nthreads=4,
			quad_decimate=1.0,
			refine_edges=1,
		)
		self.detector = apriltag.Detector(options)
		self.AprilTags = {} #This dict will hold the ID and the the distance from the robot

	def detect(self, image):
		# process the image to get the AprilTag
		buf_as_np_array = np.frombuffer(image, np.uint8)
		rgb = buf_as_np_array.reshape((self.height, self.width, 4))
		gray = cv2.cvtColor(rgb, cv2.COLOR_BGRA2GRAY)
		results = self.detector.detect(gray)
		
		# Clear previous detections
		self.AprilTags = {}
		
		for r in results:
			# Initially store the image coordinates
			self.AprilTags[r.tag_id] = {"center": (r.center[0], r.center[1])}
		
		if len(results) > 0:
			# Get the camera position and orientation
			tag_transforms = self.get_apriltag_transform(gray, results)
			
			# Update with 3D position info
			for tag_id, transform in tag_transforms.items():
				# Extract the position vector (translation component of the transform)
				position = transform[:3, 3]
				
				# Calculate Euclidean distance from camera to tag
				distance = np.linalg.norm(position)
				
				# Update the dictionary with position and distance
				if tag_id in self.AprilTags:
					self.AprilTags[tag_id]["position"] = position
					self.AprilTags[tag_id]["distance"] = distance
					print(f"Tag ID: {tag_id}, Distance: {distance:.3f}m, Position: {position}")
		
		return self.AprilTags
	def draw(self, image):
		# draw the tags on the image
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		results = self.detector.detect(gray)
		for r in results:
			corners = r.corners.astype(np.int32).reshape((-1, 1, 2))
			cv2.polylines(gray, [corners], True, (0,255,0), 2)
			cv2.putText(gray, str(r.tag_id), (int(r.center[0]), int(r.center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		cv2.imshow("AprilTags", gray)
		cv2.waitKey(1)
		return

	def get_apriltag_transform(self, image, tag_results, tag_size=0.6):
		"""
		Calculate the homogeneous transformation matrix for detected AprilTags.
		
		Args:
			image: The source image
			tag_results: Results from the apriltag detector
			camera_matrix: 3x3 camera intrinsic matrix
			camera_position: [x, y, z] position of the camera in world coordinates
			camera_orientation: Rotation matrix or quaternion describing camera orientation
			tag_size: Size of the AprilTag in meters (default: 0.6m)
			
		Returns:
			Dictionary mapping tag IDs to their homogeneous transformation matrices
		"""
		# show the image
		cv2.imshow("AprilTags", image)
		cv2.waitKey(1)
		fx, fy = self.focal_length, self.focal_length
		# fx, fy = 800,800

		cx, cy = self.width / 2, self.height / 2
		camera_matrix = np.array([[fx, 0, cx], 
                         [0, fy, cy], 
                         [0, 0, 1]], dtype=np.float32)

		camera_to_world = np.array([
			[1, 0, 0, 0.0],
			[0, 1, 0, 0],
			[0, 0, 1, 0.3],
			[0, 0, 0, 1]
		], dtype=np.float32) 

		# print(f"Camera to world: \n{camera_to_world.shape}")
		# We assume no distortion for simplicity
		distortion_coeffs = np.zeros(5)
		
		tag_transforms = {}

		for tag in tag_results:
			corners = tag.corners

			# Define the 3D points of the tag in the tag's local coordinate system
			# AprilTag's origin is at its center
			object_points = np.array([
				[-tag_size/2, -tag_size/2, 0],  # Bottom-left
				[tag_size/2, -tag_size/2, 0],   # Bottom-right
				[tag_size/2, tag_size/2, 0],    # Top-right
				[-tag_size/2, tag_size/2, 0]    # Top-left
			])

			# The corners in the image
			image_points = np.array(corners, dtype=np.float32)

			# Get the pose of the tag relative to the camera
			success, rvec, tvec = cv2.solvePnP(
				object_points,
				image_points,
				camera_matrix,
				distortion_coeffs
			)

			if not success:
				continue

			# Convert rotation vector to rotation matrix
			rotation_matrix, _ = cv2.Rodrigues(rvec)

			# Create tag-to-camera transformation matrix
			tag_to_camera = np.eye(4)
			tag_to_camera[:3, :3] = rotation_matrix
			tag_to_camera[:3, 3] = tvec.flatten()

			# Calculate tag-to-world transformation matrix
			tag_to_world = camera_to_world @ tag_to_camera
			tag_transforms[tag.tag_id] = tag_to_world

		return tag_transforms