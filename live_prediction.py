import mediapipe as mp
import cv2
from tensorflow.keras.models import load_model
from capturing_poses import extract_keypoints, mediapipe_detection
import numpy as np
import json

MODEL_PATH = "best_dance_model.keras"
LABEL_MAP_PATH = "label_map.json"
HEIGHT = 540
WIDTH = 960
FPS = 30

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

index_to_label = {int(index): label for label, index in label_map.items()}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

sequence = []
threshold = 0.4

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

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
model = load_model(MODEL_PATH
                   )
with mp_holistic.Holistic( min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        image, results = mediapipe_detection(frame, holistic)
        print(results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-120:]

        if len(sequence) == 120:
            input_sequence = np.expand_dims(sequence, axis=0)
            res = model.predict(input_sequence, verbose=0)[0]

            predicted_index = np.argmax(res)
            predicted_label = index_to_label[predicted_index]
            confidence = res[predicted_index]

            if confidence > threshold:
                cv2.putText(image,f"{predicted_label}: {confidence:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                print(predicted_label, confidence)

        cv2.imshow('Dance Prediction', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

    

