import cv2
import numpy as np
import dt_apriltags

def detect_apriltag(image_path):
	# Load image
	image = cv2.imread(image_path)
	if image is None:
		print(f"Error: Could not load image from {image_path}")
		return

	# turn image from png to jpg
	cv2.imwrite('temp.jpg', image)
	image = cv2.imread('temp.jpg')
	
	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Initialize AprilTag detector for tag36h11 family
	detector = dt_apriltags.Detector(
		families='tag36h11',
		nthreads=4,
		quad_decimate=1.0,
		refine_edges=1,
	)
	
	camera_params = [800,800, 0, 0]  # Focal length and principal point
	result = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=600)
	print(f"Detected {len(result)} AprilTags")
	print(result[0].pose_R)
	
def detect_webcam():
	cap = cv2.VideoCapture(0)
	
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		results = detector.detect(gray)

		for r in results:
			cv2.polylines(frame, [r.corners.astype(int)], True, (0,255,0), 2)

		cv2.imshow("AprilTags", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

def main():
	# Use either a file path or 0 for webcam
	detect_apriltag("/home/nick/Documents/School/Semester_8_Projects/Robotic-Systems-1/apriltags36h11_100by100/tag36h11_100_1.png")  # Replace with your image path
	# detect_webcam()  # Uncomment to use webcam
if __name__ == "__main__":
	main()
	