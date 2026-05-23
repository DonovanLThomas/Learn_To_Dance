import cv2
import numpy as np
import os
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

VIDEO_DIR = "/Users/dono_1k/Learn_To_Dance/dance_clips"
POSE_DIR = "/Users/dono_1k/Learn_To_Dance/pose_data"


def build_video_dict(video_dir):
    video_dict = {}

    for folder in os.listdir(video_dir):
        folder_path = os.path.join(video_dir, folder)

        # Skip random files like .DS_Store
        if not os.path.isdir(folder_path):
            continue

        video_dict[folder] = []

        for file_name in sorted(os.listdir(folder_path)):
            if file_name.lower().endswith(".mp4"):
                video_dict[folder].append(file_name)

    return video_dict


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = model.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results


def extract_keypoints(results):
    if results.pose_landmarks:
        pose = []

        for landmark in results.pose_landmarks.landmark:
            pose.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])

        return np.array(pose)

    else:
        return np.zeros(33 * 4)


def process_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    sequence = []
    frame_count = 0
    pose_detected_count = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            image, results = mediapipe_detection(frame, holistic)

            if results.pose_landmarks:
                pose_detected_count += 1

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # Optional preview while processing
            cv2.imshow("Pose Extraction Preview", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()

    sequence = np.array(sequence)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, sequence)

    print("-----------------------------------")
    print(f"Video: {video_path}")
    print(f"Saved: {save_path}")
    print(f"Frames processed: {frame_count}")
    print(f"Pose detected frames: {pose_detected_count}/{frame_count}")
    print(f"Saved array shape: {sequence.shape}")


def main():
    video_dict = build_video_dict(VIDEO_DIR)

    print("Found dance classes:")
    for dance_type, video_files in video_dict.items():
        print(f"{dance_type}: {len(video_files)} videos")

    for dance_type, video_files in video_dict.items():
        for video_file in video_files:
            video_path = os.path.join(VIDEO_DIR, dance_type, video_file)

            file_base_name = os.path.splitext(video_file)[0]
            save_path = os.path.join(POSE_DIR, dance_type, file_base_name + ".npy")

            print(f"Processing {dance_type}: {video_file}")

            process_video(video_path, save_path)

    cv2.destroyAllWindows()
    print("Finished extracting pose data.")


if __name__ == "__main__":
    main()