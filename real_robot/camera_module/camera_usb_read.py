import cv2


def list_cameras(max_index=10):
    """
    Scan and return a list of available camera indices.

    Args:
        max_index (int): Maximum index to probe (inclusive).

    Returns:
        List[int]: List of indices for cameras that can be opened.
    """
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


def main():
    # List all available USB cameras
    print("Scanning for available cameras...")
    cams = list_cameras(max_index=10)
    if not cams:
        print("No cameras found.")
        return

    print(f"Available camera indices: {cams}")
    cam_index = cams[-1]
    print(f"Opening camera at index {cam_index}...")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera at index {cam_index}")
        return

    # Disable autofocus (if supported by your camera/driver)
    if cap.set(cv2.CAP_PROP_AUTOFOCUS, 0):
        print("Autofocus disabled.")
    else:
        print("Warning: Could not disable autofocus; your camera may not support this property.")

    # Optionally, set a manual focus value (0.0 - 1.0)
    # focus_value = 0.0
    # cap.set(cv2.CAP_PROP_FOCUS, focus_value)

    print("Press 'q' in the window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        cv2.imshow(f'USB Camera {cam_index}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
