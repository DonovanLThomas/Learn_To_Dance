import mediapipe as mp
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from capturing_poses import extract_keypoints, mediapipe_detection
import numpy as np
import json

MODEL_PATH = "best_dance_model.weights.h5"
LABEL_MAP_PATH = "label_map.json"

def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

index_to_label = {int(index): label for label, index in label_map.items()}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

sequence = []
threshold = 0.6
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1640,
    capture_height=1232,
    display_width=820,
    display_height=616,
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

WIDTH = 820
HEIGHT = 616
FPS = 30

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

num_classes = len(label_map)
model = build_model((120, 132), num_classes)
model.load_weights(MODEL_PATH)

with mp_holistic.Holistic( min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        image, results = mediapipe_detection(frame, holistic)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-120:]

        if len(sequence) == 120:
            input_sequence = np.expand_dims(sequence, axis=0)
            res = model.predict(input_sequence, verbose=0)[0]

            predicted_index = np.argmax(res)
            predicted_label = index_to_label[predicted_index]
            confidence = res[predicted_index]

            for i, prob in enumerate(res):
                print(index_to_label[i], round(float(prob), 3))

            if confidence > threshold:
                cv2.putText(image,f"{predicted_label}: {confidence:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                print(predicted_label, confidence)

        cv2.imshow('Dance Prediction', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

    
