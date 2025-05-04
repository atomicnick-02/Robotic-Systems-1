
import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 25, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

image = "/home/nick/Documents/School/Semester_8_Projects/Robotic-Systems-1/controllers/my_first_controller0/source/test_april/image.png"
# image = "/home/nick/Documents/School/Semester_8_Projects/Robotic-Systems-1/controllers/my_first_controller0/source/test_april/chessA4_25mm.png"
img = cv.imread(image)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (10,7), None)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)
    print("WHAAAA")
    # Draw and display the corners
    cv.drawChessboardCorners(img, (10,7), corners2, ret)
    cv.imshow('img', img)
    cv.waitKey(50000)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("ret: ", ret)
    fx = mtx[0][0]
    fy = mtx[1][1]
    cx = mtx[0][2]
    cy = mtx[1][2]
    print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
cv.destroyAllWindows()

# import numpy as np
# import cv2
# from aprilgrid import Detector
# from glob import glob

# if __name__ == '__main__':
#     file_list = sorted(glob("/home/nick/Documents/School/Semester_8_Projects/Robotic-Systems-1/apriltags36h11_100by100/80_80cm.png"))
#     detector = Detector('t36h11')
#     count = 0
#     for i, file_name in enumerate(file_list):
#         img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
#         img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#         detections = detector.detect(img)
#         print(f"frame {i}, detect {len(detections)} tags")
#         count += len(detections)
#         for detection in detections:
#             center = np.round(np.average(detection.corners, axis=0)).astype(np.int32)
#             cv2.putText(img_color, f"{detection.tag_id}", center[0], 2, 2, (0, 0, 255))
#             for j, c in enumerate(detection.corners):
#                 c = np.round(c[0]).astype(np.int32)
#                 id = detection.tag_id*4+j
#                 cv2.putText(img_color, f"{id}", c, 2, 2, (0, 0, 255))
#                 cv2.circle(img_color, c, 3, (0, 255, 0))


#         img_color = cv2.resize(img_color, None, None, 0.5, 0.5)
#         cv2.imshow("im", img_color)
#         cv2.waitKey(0)
#     print(f"avg: {count/len(file_list):.3f} tags")