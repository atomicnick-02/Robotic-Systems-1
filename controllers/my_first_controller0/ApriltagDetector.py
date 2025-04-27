import cv2
import numpy as np
import apriltag

class AprilTagDetector:
	def __init__(self, camera):
		self.camera = camera
		self.width = camera.getWidth()
		self.height = camera.getHeight()
		self.focal_length = camera.getFocalLength()
		options = apriltag.DetectorOptions(
			families='tag36h11',
			nthreads=4,
			quad_decimate=1.0,
			refine_edges=1,
		)
		self.detector = apriltag.Detector(options)
		self.AprilTags = {} #This dict will hold the ID and the the distance from the robot

	def detect(self,image):
		# process the image to get the AprilTag
		buf_as_np_array = np.frombuffer(image, np.uint8)
		rgb = buf_as_np_array.reshape((self.height, self.width, 4))
		gray = cv2.cvtColor(rgb, cv2.COLOR_BGRA2GRAY)
		results = self.detector.detect(gray)
		for r in results:
			self.AprilTags[r.tag_id] = (r.center[0], r.center[1])
		# draw the tags on the image
		# self.draw(gray)
		self.calculate_distance(results, gray)
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
	def calculate_distance(self, results, image):
		fx = self.focal_length #focal length in pixels
		fy = self.focal_length
		fx, fy = 800, 800 # Example focal length in pixels
		# Assuming square pixels, fx = fy
		cx = self.width / 2
		cy = self.height / 2
		k1 = 0.1
		k2 = 0.1
		p1 = 0.0
		p2 = 0.0
		k3 = 0.0
		# Load your camera calibration parameters
		# These are crucial for accurate pose estimation
		camera_matrix = np.array([[fx, 0, cx], 
								[0, fy, cy], 
								[0, 0, 1]], dtype=np.float32)
		distortion_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

		# Tag size in meters (measure your physical tag)
		tag_size = 0.6  #in m


		for r in results:
			# Extract tag ID and corner positions
			tag_id = r.tag_id
			corners = r.corners
			
			# Get the homography matrix
			H = r.homography
			
			# Use OpenCV's solvePnP to get the rotation and translation vectors
			object_points = np.array([
				[-tag_size/2, -tag_size/2, 0],
				[ tag_size/2, -tag_size/2, 0],
				[ tag_size/2,  tag_size/2, 0],
				[-tag_size/2,  tag_size/2, 0]
			])
			
			image_points = np.array(corners, dtype=np.float32)
			
			success, rvec, tvec = cv2.solvePnP(
				object_points, 
				image_points,
				camera_matrix,
				distortion_coeffs
			)
			
			# tvec contains the position information (x, y, z) in camera coordinates
			position = tvec.flatten()
			print(f"Tag ID: {tag_id}, Position: {position}")
			
			# Draw axis on the tag (optional visualization)
			cv2.drawFrameAxes(image, camera_matrix, distortion_coeffs, rvec, tvec, tag_size)
			cv2.imshow("AprilTag Pose", image)	
			cv2.waitKey(1)