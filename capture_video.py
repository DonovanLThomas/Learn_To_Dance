import numpy as np
import cv2
import os
import subprocess
import time

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=6,
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

HEIGHT = 540
WIDTH = 960
FPS = 30

os.makedirs("clips", exist_ok=True)

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Camera failed to open")
    exit()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

recording = False
out = None
last_saved_clip = None

print("Controls:")
print("r = start recording")
print("s = stop and save")
print("d = delete last saved clip")
print("q = quit")


while(True):
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    if recording:
        out.write(frame)
        cv2.putText(
            frame,
            "REC",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    cv2.imshow("Dance Clip Recorder", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r') and not recording:

        filename = f"clips/dance_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, fourcc, FPS, (WIDTH, HEIGHT))

        if not out.isOpened():
            print("Failed to open video writer")
            out = None
        else:
            recording = True
            print(f"Recording started: {filename}")

    elif key == ord("s") and recording:
        recording = False
        if out is not None:
            out.release()
            out = None
        last_saved_clip = filename
        print(f"Recording stopped and saved: {last_saved_clip}")

    elif key == ord("d") and not recording:
        if last_saved_clip is not None and os.path.exists(last_saved_clip):
            os.remove(last_saved_clip)
            print(f"Deleted last clip: {last_saved_clip}")
            last_saved_clip = None
        else:
            print("No saved clip to delete")
        
    elif key == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()