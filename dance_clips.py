import os
import cv2
import time

# Collecting mp4 videos for pose estimation

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1640
    ,capture_height=1232
    ,display_width=820
    ,display_height=616
    ,framerate=30,
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

WIDTH = 820
HEIGHT = 616
FPS = 30

COUNTDOWN_SECONDS = 2
CLIP_SECONDS = 4
MAX_CLIPS = 40

dances = ["idle", "dab", "milly-rock", "dougie"]

for dance in dances:
    os.makedirs(f"dance_clips/{dance}", exist_ok=True)

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

dance_pos = 0
clip_number = 1

recording = False
counting_down = False
collecting = False

countdown_start_time = None
recording_start_time = None

out = None
filename = None

print("Controls:")
print("r = start automatic recording")
print("q = quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    current_dance = dances[dance_pos]

    # Always show current status
    cv2.putText(
        frame,
        f"Dance: {current_dance}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Clip: {clip_number}/{MAX_CLIPS}",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Countdown mode
    if counting_down:
        elapsed_countdown = time.time() - countdown_start_time
        remaining = COUNTDOWN_SECONDS - int(elapsed_countdown)

        if remaining > 0:
            cv2.putText(
                frame,
                str(remaining),
                (WIDTH // 2 - 30, HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                (0, 255, 255),
                8
            )
        else:
            counting_down = False
            recording = True
            recording_start_time = time.time()

            filename = f"dance_clips/{current_dance}/{current_dance}_{clip_number}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(filename, fourcc, FPS, (WIDTH, HEIGHT))

            if not out.isOpened():
                print("Failed to open video writer")
                out = None
                recording = False
                collecting = False
            else:
                print(f"Recording started: {filename}")

    # Recording mode
    if recording:
        out.write(frame)

        elapsed_recording = time.time() - recording_start_time
        remaining_recording = CLIP_SECONDS - elapsed_recording

        cv2.putText(
            frame,
            "REC",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            f"{remaining_recording:.1f}s",
            (WIDTH - 180, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        if elapsed_recording >= CLIP_SECONDS:
            recording = False

            if out is not None:
                out.release()
                out = None

            print(f"Saved clip: {filename}")

            if clip_number >= MAX_CLIPS:
                dance_pos += 1
                clip_number = 1

                if dance_pos >= len(dances):
                    print("Finished collecting all dance clips.")
                    break

                print(f"Switching to next dance: {dances[dance_pos]}")
            else:
                clip_number += 1

            # Automatically start next countdown
            if collecting:
                counting_down = True
                countdown_start_time = time.time()

    cv2.imshow("Dance Clip Recorder", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r") and not collecting:
        collecting = True
        counting_down = True
        countdown_start_time = time.time()
        print("Starting collection...")

    elif key == ord("q"):
        break

cap.release()

if out is not None:
    out.release()

cv2.destroyAllWindows()