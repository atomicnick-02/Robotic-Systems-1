#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import pupil_apriltags as apriltag
import glob
from pathlib import Path

def detect_apriltags(image_path):
    """
    Detect AprilTags in the given image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        detections: List of AprilTag detections
        image: The loaded image
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create detector (36h11 family)
    detector = apriltag.Detector(families='tag36h11',
                                nthreads=4,
                                quad_decimate=1.0,
                                quad_sigma=0.0,
                                refine_edges=1,
                                decode_sharpening=0.25,
                                debug=0)
    
    # Detect AprilTags
    detections = detector.detect(gray)
    print(f"Detected {len(detections)} tags in {image_path}")
    
    return detections, image

def draw_detections(image, detections):
    """Draw the detected AprilTags on the image"""
    # Make a copy of the image to draw on
    result = image.copy()
    
    for det in detections:
        # Extract corners and convert to integers
        corners = np.array(det.corners, dtype=np.int32).reshape((-1, 1, 2))
        
        # Draw the tag outline
        cv2.polylines(result, [corners], True, (0, 255, 0), 2)
        
        # Draw the tag center
        center = tuple(np.mean(corners, axis=0)[0].astype(int))
        cv2.circle(result, center, 5, (0, 0, 255), -1)
        
        # Print tag ID
        cv2.putText(result, str(det.tag_id), 
                    (center[0] - 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return result

def create_apriltag_board_points(tag_size=0.08, tag_spacing=0.02, rows=6, cols=6):
    """
    Create the 3D points for an AprilGrid calibration board
    
    Args:
        tag_size: Size of each tag in meters (default 8cm)
        tag_spacing: Spacing between tags in meters (default 2cm)
        rows: Number of tag rows (default 6)
        cols: Number of tag columns (default 6)
        
    Returns:
        board_points: Dictionary mapping tag IDs to their 3D coordinates
    """
    board_points = {}
    
    for row in range(rows):
        for col in range(cols):
            tag_id = row * cols + col
            
            # Calculate the tag position in the grid
            # Bottom-left corner of the tag
            x0 = col * (tag_size + tag_spacing)
            y0 = row * (tag_size + tag_spacing)
            
            # All four corners of the tag
            corners = [
                [x0, y0, 0],                      # Bottom-left
                [x0 + tag_size, y0, 0],           # Bottom-right
                [x0 + tag_size, y0 + tag_size, 0], # Top-right
                [x0, y0 + tag_size, 0]            # Top-left
            ]
            
            board_points[tag_id] = np.array(corners)
            
    return board_points

def calibrate_camera_from_apriltags(image_paths, tag_size=0.08, tag_spacing=0.02, rows=6, cols=6):
    """
    Calibrate camera using AprilTag grid
    
    Args:
        image_paths: List of paths to calibration images
        tag_size: Size of each tag in meters (default 8cm)
        tag_spacing: Spacing between tags in meters (default 2cm)
        rows: Number of tag rows (default 6)
        cols: Number of tag columns (default 6)
        
    Returns:
        ret: RMS reprojection error
        mtx: Camera matrix (intrinsic parameters)
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    # Create 3D points for the AprilGrid
    board_points = create_apriltag_board_points(tag_size, tag_spacing, rows, cols)
    
    # Lists to store object points and image points from all images
    objpoints = []  # 3D points in world space
    imgpoints = []  # 2D points in image plane
    
    # Process each image
    for image_path in image_paths:
        try:
            # Detect AprilTags
            detections, image = detect_apriltags(image_path)
            
            if not detections:
                print(f"No AprilTags detected in {image_path}")
                continue
            
            # Get image size
            img_height, img_width = image.shape[:2]
            
            # Prepare object and image points for this image
            obj_points_this_img = []
            img_points_this_img = []
            
            # For each detected tag
            for det in detections:
                tag_id = det.tag_id
                
                # Check if this tag ID is in our board model
                if tag_id in board_points:
                    # Add the 3D points for this tag
                    obj_points_this_img.extend(board_points[tag_id])
                    
                    # Add the corresponding 2D points (detected corners)
                    img_points_this_img.extend(det.corners)
            
            # If we have enough points, add them to our lists
            if len(obj_points_this_img) >= 4:  # need at least 4 points
                objpoints.append(np.array(obj_points_this_img, dtype=np.float32))
                imgpoints.append(np.array(img_points_this_img, dtype=np.float32))
                print(f"Added {len(obj_points_this_img)} points from {image_path}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    if not objpoints:
        raise ValueError("No valid calibration points found in any image")
    
    # Calibrate camera
    print(f"Calibrating camera with {len(objpoints)} images...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (img_width, img_height), None, None)
    
    return ret, mtx, dist, rvecs, tvecs

def save_calibration(output_file, ret, mtx, dist):
    """Save calibration results to a file"""
    np.savez(output_file, 
             camera_matrix=mtx,
             dist_coeffs=dist, 
             rms_error=ret)
    
    # Also save in a more human-readable format
    with open(f"{output_file.split('.')[0]}.txt", 'w') as f:
        f.write(f"RMS Reprojection Error: {ret}\n\n")
        f.write(f"Camera Matrix:\n{mtx}\n\n")
        f.write(f"Distortion Coefficients:\n{dist}\n")
    
    print(f"Calibration saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Camera calibration using AprilGrid')
    path = "/home/nick/Documents/School/Semester_8_Projects/Robotic-Systems-1/controllers/my_first_controller0/source/test_april/image.png"
    output_path = "/home/nick/Documents/School/Semester_8_Projects/Robotic-Systems-1/controllers/my_first_controller0/source/test_april/output.png"
    tag_size = 0.088
    tag_spacing = 0.0264
    rows = 6
    cols = 6


    image_paths = [path]
    tag_size = 0.088
    tag_spacing = 0.0264
    rows = 6
    cols = 6
    output_path = output_path
    
    # Visualize detections if requested
    if True:
        for img_path in image_paths:
            detections, image = detect_apriltags(img_path)
            if detections:
                result = draw_detections(image, detections)
                
                # Resize large images for display
                h, w = result.shape[:2]
                scale = min(1.0, 1200/max(h, w))
                if scale < 1.0:
                    new_h, new_w = int(h*scale), int(w*scale)
                    result = cv2.resize(result, (new_w, new_h))
                
                # Show the result
                cv2.imshow('AprilTag Detections', result)
                key = cv2.waitKey(0)
                if key == 27:  # ESC key
                    break
        cv2.destroyAllWindows()
    
    # Perform calibration
    ret, mtx, dist, rvecs, tvecs = calibrate_camera_from_apriltags(
        image_paths, 
        tag_size=tag_size,
        tag_spacing=tag_spacing,
        rows=rows,
        cols=cols
    )
    
    # Print calibration results
    print("\nCalibration Results:")
    print(f"RMS Reprojection Error: {ret}")
    print(f"\nCamera Matrix:\n{mtx}")
    print(f"\nDistortion Coefficients:\n{dist}")
    
    # Save calibration
    save_calibration(f"{output_path}", ret, mtx, dist)

if __name__ == "__main__":
    main()