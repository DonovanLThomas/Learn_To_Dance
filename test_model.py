import numpy as np
import json
from tensorflow.keras.models import load_model
import os

MODEL_PATH = "best_dance_model.keras"
LABEL_MAP_PATH = "label_map.json"
SEQUENCE_LENGTH = 120
DATA_PATH = "pose_data"

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

index_to_label = {index: label for label, index in label_map.items()}

def prepare_sequence(sample_path):
    pose_sequence = np.load(sample_path)

    if len(pose_sequence) > SEQUENCE_LENGTH:
        pose_sequence = pose_sequence[:SEQUENCE_LENGTH]

    elif len(pose_sequence) < SEQUENCE_LENGTH:
        missing_frames = SEQUENCE_LENGTH - len(pose_sequence)
        num_features = pose_sequence.shape[1]
        padding = np.zeros((missing_frames, num_features))
        pose_sequence = np.vstack([pose_sequence, padding])

    pose_sequence = np.expand_dims(pose_sequence, axis=0)
    return pose_sequence
model = load_model(MODEL_PATH)

correct = 0
total = 0

for folder in label_map.keys():
    folder_path = os.path.join(DATA_PATH, folder)

    for file in sorted(os.listdir(folder_path)): 
        if not file.endswith(".npy"):
            continue

        sample_path = os.path.join(folder_path, file)
        pose_sequence = prepare_sequence(sample_path)

        prediction = model.predict(pose_sequence, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_label = index_to_label[predicted_index]

        if predicted_label != folder:
            print(file, "true:", folder, "predicted:", predicted_label)
        else:
            correct += 1

        total += 1

print("Accuracy:", correct / total)
print("Correct:", correct)
print("Total:", total)
        