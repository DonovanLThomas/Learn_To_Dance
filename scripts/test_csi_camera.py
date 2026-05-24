import cv2
import sys


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

print("Python:", sys.executable)
print("OpenCV:", cv2.__version__)
print("cv2 path:", cv2.__file__)
print("Pipeline:", gstreamer_pipeline())

for line in cv2.getBuildInformation().splitlines():
    if "GStreamer" in line:
        print(line)

if not cap.isOpened():
    print("Camera failed to open")
    raise SystemExit(1)

print("Camera opened. Press q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("CSI Camera Test", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
