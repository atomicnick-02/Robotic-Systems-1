import cv2

# Built a class that i will use for detecting april tags
class AprilTagDetector:
	def __init__(self):
		self.detector = cv2.aruco.ArucoDetector()

	def detect(self, image):
		return self.detector.detectMarkers(image)

	def draw(self, image, corners, ids):
		return cv2.aruco.drawDetectedMarkers(image, corners, ids)
	
	