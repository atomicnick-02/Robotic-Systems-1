import cv2
import numpy as np
import argparse
from collections import defaultdict

# Optional: pip install pupil_apriltags or apriltag
try:
    from pupil_apriltags import Detector
except ImportError:
    from apriltag import ApriltagDetector as Detector


def detect_apriltags(img_gray, tag_family="36h11", nthreads=4):
    """
    Detect AprilTags in a grayscale image.
    Returns a list of detections, each with tag_id and corner pixel points.
    """
    detector = Detector(families=tag_family, nthreads=nthreads)
    detections = detector.detect(img_gray)
    results = []
    for det in detections:
        corners = np.array(det.corners, dtype=np.float32)
        results.append({
            'id': det.tag_id,
            'corners': corners
        })
    return results


def build_object_points(detections, tag_size, tag_positions):
    """
    Given list of detections and known 3D center positions of each tag,
    build object_points and image_points lists for calibration.

    tag_size: side length of tag (float, in meters).
    tag_positions: dict mapping tag_id to (x, y, z) center in world coords.
    """
    obj_pts = []
    img_pts = []

    half = tag_size / 2.0
    # tag corner offsets from center in world frame (clockwise order matching detector)
    corner_offsets = np.array([
        [-half,  half, 0],  # top-left
        [ half,  half, 0],  # top-right
        [ half, -half, 0],  # bottom-right
        [-half, -half, 0],  # bottom-left
    ], dtype=np.float32)

    for det in detections:
        tid = det['id']
        if tid not in tag_positions:
            continue
        center = np.array(tag_positions[tid], dtype=np.float32)
        # build 4 world pts per tag
        for i in range(4):
            obj_pts.append(center + corner_offsets[i])
            img_pts.append(det['corners'][i])

    return np.array(obj_pts, dtype=np.float32), np.array(img_pts, dtype=np.float32)


def calibrate_from_image(image_path, tag_size, tag_positions, tag_family="36h11"):
    # Load and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect tags
    detections = detect_apriltags(gray, tag_family=tag_family)
    print(f"Detected {len(detections)} tags.")

    # Build correspondences
    obj_pts, img_pts = build_object_points(detections, tag_size, tag_positions)
    if len(obj_pts) < 6:
        raise ValueError("Not enough points for calibration. Need at least 6.")

    # prepare for calibrateCamera
    obj_pts_list = [obj_pts]
    img_pts_list = [img_pts]
    img_size = (img.shape[1], img.shape[0])

    # Camera calibration
    ret, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts_list, img_pts_list, img_size, None, None
    )
    print(f"Reprojection error: {ret}")
    return camera_matrix, dist_coefs, rvecs, tvecs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calibrate camera using AprilTag detections."
    )
    path = "image.png"
    tag_size = 0.6
    tag_positions = "{0:[0,0,0],1:[d,0,0]}"
    tag_family = "36h11"
    
    # Parse tag_positions
    tag_positions = eval(tag_positions)

    K, dist, rvecs, tvecs = calibrate_from_image(
        path, tag_size, tag_positions, tag_family
    )
    print("Camera matrix (K):\n", K)
    print("Distortion coefficients: ", dist.ravel())
    for i, (r, t) in enumerate(zip(rvecs, tvecs)):
        print(f"Tag {i} pose -> Rvec: {r.ravel()}, Tvec: {t.ravel()}")
