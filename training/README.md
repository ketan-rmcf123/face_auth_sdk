# Training workspace

This folder contains notebook-based training entry points for:

- covered-face classification
- anti-spoof classification

The notebooks use a lightweight PyTorch workflow and assume you have a local dataset structure under `training/data/`.

## Suggested folder layout

```text
training/
  data/
    covered_face/
      train/
      val/
    anti_spoof/
      live/
      spoof/
  notebooks/
    01_covered_face_classifier.ipynb
    02_anti_spoof_classifier.ipynb
```

## Notes

- For face detection, the notebooks use a fast MTCNN-style approach via `facenet-pytorch`.
- For the anti-spoof model, the scaffold uses a simple CNN classifier that can be replaced with a stronger representation-learning backbone later.
