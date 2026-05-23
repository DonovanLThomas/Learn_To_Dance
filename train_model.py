import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import json

DATA_PATH = "pose_data"
SEQUENCE_LENGTH = 120


def load_dataset(DATA_PATH):
    X = []
    y = []

    classes = sorted([
        folder for folder in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, folder))
    ])

    label_map = {class_name: index for index, class_name in enumerate(classes)}

    for folder in classes:

        folder_path = os.path.join(DATA_PATH, folder)

        for file in sorted(os.listdir(folder_path)):

            if file.endswith(".npy"):
                sample_path = os.path.join(folder_path, file)
                pose_sequence = np.load(sample_path)

                if len(pose_sequence) > SEQUENCE_LENGTH:
                    pose_sequence = pose_sequence[:SEQUENCE_LENGTH]

                elif len(pose_sequence) < SEQUENCE_LENGTH:
                    missing_frames = SEQUENCE_LENGTH - len(pose_sequence)
                    num_features = pose_sequence.shape[1]
                    padding = np.zeros((missing_frames, num_features))
                    pose_sequence = np.vstack([pose_sequence, padding])
                
                X.append(pose_sequence)
                y.append(label_map[folder])

    X = np.array(X)
    y = np.array(y)
    return X, y, label_map

def split_data(X, y):


    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test

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

def main():
    X, y, label_map = load_dataset(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    input_shape = (SEQUENCE_LENGTH, X.shape[2])
    num_classes = len(label_map)

    model = build_model(input_shape, num_classes)
    model.summary()

    checkpoint = ModelCheckpoint(
        "best_dance_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        callbacks=[checkpoint]
    )

    best_model = load_model("best_dance_model.keras")

    test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
    print("Best model test accuracy:", test_accuracy)

    y_pred_probs = best_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(
        y_test,
        y_pred,
        target_names=list(label_map.keys())
    ))
    
    with open("label_map.json", "w") as f:
        json.dump(label_map, f)

main()