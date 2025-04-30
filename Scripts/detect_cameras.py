import cv2

def list_cameras():
    available_cameras = []
    for index in range(2):  # Check only /dev/video0 and /dev/video1
        cap = cv2.VideoCapture(index, cv2.CAP_GSTREAMER)  # Use the GStreamer backend
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
        else:
            cap.release()
    return available_cameras

cameras = list_cameras()
if cameras:
    print(f"Available camera sources: {cameras}")
else:
    print("No cameras detected.")