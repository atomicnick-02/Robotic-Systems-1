import cv2
import numpy as np
import math
from pupil_apriltags import Detector  

class AprilTagDetector:
    """A class for detecting and processing AprilTags in images."""

    def __init__(self, camera, robot):
        """
        Initialize the AprilTag detector with camera and robot parameters.

        Args:
            camera: Camera object providing image capture capabilities
            robot: Robot object for navigation and control
        """
        self.camera = camera
        self.width = camera.getWidth()
        self.height = camera.getHeight()
        self.foc_distance = camera.getFocalDistance()
        self.focal_length = camera.getFocalLength()
        self.fov = 1.047
        self.robot = robot

        # Configure the pupil_apriltags detector with same parameters
        self.detector = Detector(
            families='tag36h11',
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.8,
            refine_edges=True,
            decode_sharpening=0.25,
            debug=False
        )

        self.april_tags = {}

    def detect(self, image) -> dict:
        """
        Detect AprilTags in the given image and estimate their poses.
        Adds realism by limiting range, field of view, and injecting noise.

        Returns:
            dict: tag_id -> list of noisy tvecs (in robot frame)
        """
        # Convert image to grayscale
        image = self.camera.getImage()
        image_array = np.frombuffer(image, np.uint8)
        rgb_image = image_array.reshape((self.height, self.width, 4))
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2GRAY)

        # Camera calibration
        fx = fy = self.width / (2 * math.tan(self.fov / 2))  
        cx = self.width / 2
        cy = self.height / 2
        camera_matrix = [fx, fy, cx, cy]

        tags = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_matrix,
            tag_size=0.128
        )

        max_range = 3.5  # meters
        max_angle = np.deg2rad(30)  # ±30° FOV

        tag_locations = {}

        for tag in tags:
            tvec = np.array(tag.pose_t)   

            # Rotate to robot frame
            tvec = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) @ tvec

            # Distance and angle filtering
            r = np.linalg.norm(tvec)
            angle = np.arctan2(tvec[1, 0], tvec[0, 0])

            if r > max_range or abs(angle) > max_angle:
                continue  # Skip out-of-range or out-of-FOV tags


            tag_id = tag.tag_id
            if tag_id not in tag_locations:
                tag_locations[tag_id] = [tvec]
            else:
                tag_locations[tag_id].append(tvec)

        # Sort each tag's detections by distance
        for tag_id, tvec_list in tag_locations.items():
            tag_locations[tag_id] = sorted(tvec_list, key=lambda t: np.linalg.norm(t))

        return tag_locations

