import numpy as np
import cv2 as cv
import glob
np.set_printoptions(suppress=True, precision=4)
import os

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 25, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Lists to store calibration results
all_mtx = []
all_dist = []
all_rvecs = []
all_tvecs = []

os.chdir('./controllers/my_first_controller0/source/test_april/')
images = glob.glob('*.png')

for fname in images:
	img = cv.imread(fname)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
	# Find the chess board corners
	ret, corners = cv.findChessboardCorners(gray, (10,7), None)
	print("ret: ", fname)
	# If found, add object points, image points (after refining them)
	if ret == True:
		objpoints.append(objp)
		corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
		imgpoints.append(corners2)
 
		# Draw and display the corners
		cv.drawChessboardCorners(img, (10,7), corners2, ret)
		ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
		
		# Store the calibration results
		all_mtx.append(mtx)
		all_dist.append(dist)
		all_rvecs.append(rvecs)
		all_tvecs.append(tvecs)

		cv.imshow('img', img)
		cv.waitKey(500)
 
cv.destroyAllWindows()

# Calculate and print average calibration parameters
if len(all_mtx) > 0:
	avg_mtx = np.mean(all_mtx, axis=0)
	avg_dist = np.mean(all_dist, axis=0)
	print("\nAverage Camera Matrix:\n", avg_mtx)
	print("\nAverage Distortion Coefficients:\n", avg_dist)