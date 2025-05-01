import cv2
import numpy as np
import dt_apriltags

class AprilTagDetector:
    def __init__(self, camera, robot):
        # Initialize camera properties
        self.camera = camera
        self.width = camera.getWidth()
        self.height = camera.getHeight()
        self.foc_distance = camera.getFocalDistance()
        self.focal_length = camera.getFocalLength()
        self.robot = robot

        # Initialize AprilTag detector
        self.detector = dt_apriltags.Detector(
            families='tag36h11',
            nthreads=4,
            quad_decimate=1.0,
            refine_edges=1,
        )
        
        # Dictionary to hold ID and distance from the robot
        self.AprilTags = {}

    def detect(self, image):
        # Convert image buffer to numpy array
        buf_as_np_array = np.frombuffer(image, np.uint8)
        rgb = buf_as_np_array.reshape((self.height, self.width, 4))
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGRA2GRAY)

        # Set camera parameters
        # fx, fy = self.foc_distance, self.foc_distance
        self.focal_length = 299  # Focal length for 2.5 m
        fx, fy = self.focal_length, self.focal_length
        cx, cy = self.width / 2, self.height / 2
        camera_params = [fx, fy, cx, cy]

        # Detect AprilTags
        result = self.detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=0.6)
        if len(result) == 0:
            return

        # Process multiple detected tags
        if len(result) > 1:
            tag_locs = []
            for r in result:
                # Rotate the result by 90 degrees
                r.pose_R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) @ r.pose_R
                r.pose_t = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) @ r.pose_t
                tag_locs.append((r.tag_id,np.block([[r.pose_R, r.pose_t], [0, 0, 0, 1]])))

            # Print the difference between the first two tag locations
            print(f"Tag 1: {tag_locs[0][:3, 3]} Tag 2: {tag_locs[1][:3, 3]}", end="")
            print(tag_locs[1][:3, 3] - tag_locs[0][:3, 3])
            

            return tag_locs
