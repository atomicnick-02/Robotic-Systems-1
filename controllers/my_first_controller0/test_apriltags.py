import cv2
import numpy as np
from pupil_apriltags import Detector
import argparse
import os
def detect_apriltags(image_path):
	"""
	Detect AprilTags in an image and draw the detections.
	
	Args:
		image_path: Path to the input image file
	"""
	# Load the image
	image = cv2.imread(image_path)
	if image is None:
		print(f"Error: Could not load image from {image_path}")
		return
		
	if len(image.shape) == 3:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	else:
		gray = image.copy()

	gray = cv2.bilateralFilter(gray, 9, 75, 75)
	# Enhance contrast
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	gray = clahe.apply(gray)

	# gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
	# gray = cv2.equalizeHist(gray)  # Improve contrast
	cv2.imshow("AprilTag Detection", gray)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# Create AprilTag detector
	# 36h11 is the tag family
	detector = Detector(
		families='tag36h11',
		nthreads = 4, 
		quad_decimate = 1.5, # [0,5]
		quad_sigma = 0.8, # [0,2]
		refine_edges = 1, # {0,1}
		decode_sharpening = 0.5, # [0,1]
		debug = 0, # {0,1}
		# max_hamming = 2, # {0,1,2,3}
	)
	
	# Known physical size of the tag in meters
	tag_size = 1  # 100mm = 0.1m
	
	# Approximate camera parameters (replace with your actual camera calibration)
	# For a webcam, these are reasonable defaults
	image_width = image.shape[1]
	image_height = image.shape[0]
	fx = image_width  # rough approximation for focal length
	fy = image_width
	cx = image_width / 2
	cy = image_height / 2
	camera_params = [fx, fy, cx, cy]
	
	# Detect AprilTags in the image
	results = detector.detect(gray, estimate_tag_pose=True, 
							 camera_params=camera_params, tag_size=tag_size)
	
	print(f"Detected {len(results)} tags")
	
	# Draw detection results
	for result in results:
		# Extract tag information
		tag_id = result.tag_id
		center = (int(result.center[0]), int(result.center[1]))
		corners = result.corners.astype(int)
		
		# Draw tag outline and ID
		cv2.polylines(image, [corners.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
		cv2.putText(image, f"ID: {tag_id}", (center[0] - 10, center[1] - 10),
				   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		
		# If pose estimation is available
		if result.pose_t is not None and result.pose_R is not None:
			# Extract distance information (z-coordinate in tag frame)
			distance = result.pose_t[2][0]
			cv2.putText(image, f"Distance: {distance:.2f}m", 
					   (center[0] - 10, center[1] + 10),
					   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
			
			# Draw axes on the tag
			axis_length = tag_size / 2
			corners3d = np.array([
				[-axis_length, -axis_length, 0],
				[ axis_length, -axis_length, 0],
				[ axis_length,  axis_length, 0],
				[-axis_length,  axis_length, 0],
				[0, 0, axis_length]  # This point is for the z-axis
			])
			
			# Project 3D points to the image plane
			K = np.array([
				[fx, 0, cx],
				[0, fy, cy],
				[0, 0, 1]
			])
			
			# Transform points to camera frame
			points_camera = result.pose_R @ corners3d.T + result.pose_t
			
			# Project to image
			projected_points = K @ points_camera
			projected_points = projected_points / projected_points[2]
			projected_points = projected_points[:2].T.astype(int)
			
			# Draw axes
			origin = tuple(projected_points[0])
			cv2.line(image, origin, tuple(projected_points[1]), (0, 0, 255), 2)  # X-axis (red)
			cv2.line(image, origin, tuple(projected_points[3]), (0, 255, 0), 2)  # Y-axis (green)
			cv2.line(image, origin, tuple(projected_points[4]), (255, 0, 0), 2)  # Z-axis (blue)
	
	# Display the result
	
	
	# Save the result
	output_path = "output_apriltag_detection.jpg"
	cv2.imwrite(output_path, image)
	print(f"Detection result saved to {output_path}")

if __name__ == "__main__":
	path = "apriltags36h11_100by100/tag36_11_00001.png"
	detect_apriltags(path)
	
	path = "apriltags36h11_100by100/pillar1.png"
	detect_apriltags(path)
	
	path = "apriltags36h11_100by100/first_sample.png"
	detect_apriltags(path)