import cv2
import numpy as np
import apriltag

def detect_apriltag(image_path):
	# Load image
	image = cv2.imread(image_path)
	if image is None:
		print(f"Error: Could not load image from {image_path}")
		return

	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Initialize AprilTag detector for tag36h11 family
	detector = apriltag.DetectorOptions(
		families='tag36h11',
		nthreads=4,
		quad_decimate=1.0,
		quad_sigma=0.6,
		refine_edges=1,
		decode_sharpening=0.25,
		hamming_dist=2  # Allows up to 2 bits of error
	)

	# Detect tags
	results = detector.detect(gray)

	# Process detections
	if not results:
		print("No AprilTags detected")
		return

	print(f"Found {len(results)} tags")
	
	# Draw detections
	for result in results:
		# Extract detection information
		tag_id = result.tag_id
		corners = result.corners.astype(int)
		center = result.center.astype(int)

		# Draw polygon around the tag
		cv2.polylines(image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
		
		# Draw center point
		cv2.circle(image, tuple(center), radius=5, color=(0, 0, 255), thickness=-1)
		
		# Add text label
		cv2.putText(image, f"ID: {tag_id}", (center[0] - 20, center[1] - 20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

	# Display results
	cv2.imshow("AprilTag Detection", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
def detect_webcam():
	cap = cv2.VideoCapture(0)
	options = apriltag.DetectorOptions(families='tag36h11')
	detector = apriltag.Detector()
	
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
if __name__ == "__main__":
	# Use either a file path or 0 for webcam
	# detect_apriltag("your_image.jpg")  # Replace with your image path
	detect_webcam()  # Uncomment to use webcam
def main():
	path = "apriltags36h11_100by100/tag36_11_00001.png"
	detect_apriltag(path)


	path = "apriltags36h11_100by100/pillar1.png"
	detect_apriltag(path)
	
		
	path = "apriltags36h11_100by100/first_sample.png"
	detect_apriltag(path)
	

if __name__ == "__main__":
	main()
	