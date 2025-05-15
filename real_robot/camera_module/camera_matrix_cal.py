import cv2
import numpy as np

# termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.001)

# prepare object points for a 10x7 chessboard
objp = np.zeros((10 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)

# storage for calibration points
objpoints = []  # 3d points in real world space\Nimgpoints = []  # 2d points in image plane
imgpoints = []  # 2d points in image plane

def list_cameras(max_index=10):
    """
    Scan indices 0..max_index for available cameras.
    """
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


def main():
    print("Scanning for available cameras...")
    cams = list_cameras(max_index=10)
    if not cams:
        print("No cameras found.")
        return

    cam_index = cams[-1]
    print(f"Opening camera at index {cam_index}...")
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera at index {cam_index}")
        return

    # disable autofocus if supported
    if cap.set(cv2.CAP_PROP_AUTOFOCUS, 0):
        print("Autofocus disabled.")
    else:
        print("Warning: Unable to disable autofocus on this device.")

    print("Press 'c' to capture a calibration frame, 'q' to quit and compute calibration.")
    gray = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        cv2.imshow('Calibration Feed', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (10, 7), None)
            if found:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
                print(f"Captured frame {len(imgpoints)} for calibration.")
                cv2.drawChessboardCorners(frame, (10, 7), corners2, found)
                cv2.imshow('Chessboard', frame)
                cv2.waitKey(500)
                cv2.destroyAllWindows()
            else:
                print("Chessboard pattern not detected. Try again.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not imgpoints:
        print("No calibration frames collected. Exiting.")
        return

    # perform calibration
    ret_val, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    print("\nCalibration Results:")
    print(f"Re-projection error: {ret_val}")
    print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist.ravel())
    print("Rotation vectors:\n", rvecs)
    print("Translation vectors:\n", tvecs)

    # save parameters to file
    np.savez('camera_calibration.npz', 
             ret=ret_val, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("Calibration data saved to 'camera_calibration.npz'.")


if __name__ == '__main__':
    main()
