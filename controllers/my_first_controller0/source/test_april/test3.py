import numpy as np
import cv2 as cv
import glob
import os

np.set_printoptions(suppress=True, precision=4)

# Termination criteria for sub-pixel corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Chessboard dimensions (number of inner corners)
chessboard_size = (10, 7)  # 10 corners across columns, 7 down rows

# Square size in real-world units (e.g., millimeters)
square_size = 25  # example value for A4 printed board — adjust as needed

# Prepare object points, like (0,0,0), (1,0,0), ..., scaled by square_size
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store 3D and 2D points
objpoints = []
imgpoints = []

# Go to directory with calibration images
os.chdir('./controllers/my_first_controller0/source/test_april/')
images = glob.glob('*.png')

for fname in images:
	if fname == '80_80chess.png':
		continue
	img = cv.imread(fname)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	# Find chessboard corners
	ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

	if ret:
		objpoints.append(objp)
		corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
		imgpoints.append(corners2)

		# Draw and show corners
		cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
		cv.imshow('img', img)
		cv.waitKey(300)

cv.destroyAllWindows()

# Only calibrate if enough points were found
if len(objpoints) > 0:
	ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
		objpoints, imgpoints, gray.shape[::-1], None, None
	)

	print("\nCamera Matrix:\n", mtx)
	print("\nDistortion Coefficients:\n", dist)
	print("\nRe-projection error:")
	print(f"fx, fy = {mtx[0][0]}, {mtx[1][1]}")
	print(f"cx, cy = {mtx[0][2]}, {mtx[1][2]}")
	# Optional: compute total reprojection error
	total_error = 0
	for i in range(len(objpoints)):
		imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
		error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
		total_error += error

	print("Total error: ", total_error / len(objpoints))
else:
	print("No chessboard corners were found in any images.")
