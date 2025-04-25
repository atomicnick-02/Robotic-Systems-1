import cv2
import numpy as np
from pupil_apriltags import Detector
import os
at_detector = Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)

class AprilTagDetector:
	def __init__(self):
		# Initialize the ArUco detector properly with a dictionary
		dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
		parameters = cv2.aruco.DetectorParameters()
		self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
		self.image_width = 128
		self.image_height = 128

	def detect(self, image):
		# Check if the image needs to be converted to grayscale
		# convert image to numpy array
		
	
		# gray = self.webots_camera_to_numpy(image, self.image_width, self.image_height)		
		# Detect markers and return the results
		# res = at_detector.detect(image,estimate_tag_pose=True)
		# print(f"res {res}")
		# corners, ids, rejected = self.detector.detectMarkers(gray)
		return 
	
	def webots_camera_to_numpy(self,image_buffer, width, height):
		"""
		Convert a Webots camera image buffer to a numpy array.
		
		Args:
			image_buffer: The raw image buffer from Webots camera (BGRA format)
			width: Width of the image in pixels
			height: Height of the image in pixels
		
		Returns:
			numpy.ndarray: Image as a numpy array with shape (height, width, 3) in RGB format
		"""
		# Convert the buffer to a numpy array of uint8
		# The buffer is organized as BGRA, so we reshape to (height, width, 4)
		buffer_array = np.frombuffer(image_buffer, dtype=np.uint8).reshape(height, width, 4)
		
		# Convert from BGRA to RGB by taking only the first 3 channels and reordering them
		# Note: We're discarding the alpha channel (index 3)
		rgb_array = buffer_array[:, :, [2, 1, 0]]
		print("rgb_array", rgb_array.shape)
		# turn this into grayscale
		rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2GRAY)
		print("rgb_array grayscale", rgb_array.shape)
		print(rgb_array[:4, :4])

		return rgb_array
	
	def bgra_bytes_to_numpy(self, byte_data, width, height):
		"""
		Convert a sequence of BGRA bytes to a NumPy array.
		
		Parameters:
		-----------
		byte_data : bytes or bytearray
			The image data in BGRA format (blue, green, red, alpha bytes per pixel)
		width : int
			The width of the image in pixels
		height : int
			The height of the image in pixels
		
		Returns:
		--------
		numpy.ndarray
			A NumPy array with shape (height, width, 4) and dtype uint8
		"""
		# Ensure we have the right amount of data
		expected_size = width * height * 4
		if len(byte_data) != expected_size:
			raise ValueError(f"Expected {expected_size} bytes for a {width}x{height} BGRA image, got {len(byte_data)} bytes")
		
		# Convert bytes to a NumPy array and reshape
		np_array = np.frombuffer(byte_data, dtype=np.uint8)
		
		# Reshape to (height, width, 4) for an image with 4 channels (BGRA)
		reshaped_array = np_array.reshape((height, width, 4))
		# discard the alpha channel
		reshaped_array = reshaped_array[:,:,:3]
		
		return reshaped_array
	
	
	def draw(self, image, corners, ids):
		if ids is not None:
			return cv2.aruco.drawDetectedMarkers(image, corners, ids)
		return image

at_detector = Detector(
    families="tag36h11",
    quad_decimate=1.0,
    refine_edges=1
)


def main():
	# list current directory
	print(os.listdir())
	test_image = os.path.join(os.getcwd(), "apriltags36h11_100by100/pillar1.png")
	print(test_image)
	# show the image
	img = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
	cv2.imshow("image", img)
	cv2.waitKey(1000)
	
	# create an instance of the AprilTagDetector class
	detections = at_detector.detect(
		img, 
		estimate_tag_pose=True
	)
	#    → returns a list of Detection objects :contentReference[oaicite:1]{index=1}
	for det in detections:
		print(f"Tag ID {det.tag_id}:")
		print(f"  • Translation (t) [m]: {det.pose_t}")
		print(f"  • Rotation  (R)   [ ]:\n{det.pose_R}\n")
	else:
		print("No tags detected")
if __name__ == "__main__":
	main()