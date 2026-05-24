# Learn To Dance

Pose-based dance clip collection, pose extraction, model training, and live prediction.

## Project Layout

```text
src/
  capture_video.py       # manual video capture experiments
  dance_clips.py         # collect labeled dance clips
  capturing_poses.py     # extract MediaPipe pose arrays from clips
  train_model.py         # train the LSTM model
  test_model.py          # test trained model on pose arrays
  live_prediction.py     # run live camera prediction

scripts/
  test_csi_camera.py     # standalone Jetson CSI camera test

data/
  raw_clips/             # local recorded videos, ignored by Git
  pose_data/             # local extracted .npy pose data, ignored by Git

models/
  label_map.json         # tracked label mapping
  *.keras, *.h5          # local generated model files, ignored by Git
```

Run scripts from the repo root, for example:

```bash
python src/live_prediction.py
python src/dance_clips.py
python scripts/test_csi_camera.py
```

Personal videos, pose arrays, virtual environments, and generated model weights are ignored so `git add .` is safe for source updates.
