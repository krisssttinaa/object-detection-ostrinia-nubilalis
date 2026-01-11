# object-detection-ostrinia-nubilalis

Project for detecting Ostrinia nubilalis (European corn borer) in agricultural trap images using a Faster R-CNN object detection model.

## Project structure

Main files and folders:

- `dataset.py` — dataset loader and SVG annotation parsing
- `model_faster_rcnn.py` — Faster R-CNN model definition (ResNet-50 backbone)
- `train_model.py` — training script
- `evaluate_model.py` — evaluation on validation/test data
- `predict.py` — inference on single images
- `count_empty_svgs.py` — statistics for empty vs non-empty annotations
- `moth_detector/` — dataset directory (train / val / test splits)

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- numpy
- pillow
- matplotlib

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch torchvision numpy pillow matplotlib
```

## Dataset layout

The dataset directory should be structured as:

```
moth_detector/
  train/
  val/
  test/
```

Each split contains:
- `.jpg` images
- corresponding `.svg` annotation files

Bounding boxes for *Ostrinia nubilalis* are represented by green rectangles in SVG files.

## Count empty images (optional)

To compute statistics about empty annotations:

```bash
python count_empty_svgs.py --data_root moth_detector
```

This reports the number of images without any *Ostrinia nubilalis* instances.

## Training

Train the Faster R-CNN model:

```bash
python train_model.py \
  --data_root moth_detector \
  --epochs 10 \
  --batch_size 1 \
  --device cpu
```

Checkpoints are saved to `runs_custom_final/`, including:
- `best.pt` — best validation performance
- `last.pt` — final epoch

## Evaluation

Evaluate a trained model on the test set:

```bash
python evaluate_model.py \
  --data_root moth_detector \
  --ckpt runs_custom_final/best.pt \
  --device cpu \
  --iou 0.5 \
  --score_thr 0.7
```

Reported metrics include precision, recall, F1-score, and average precision at IoU 0.5.

## Prediction

Run on a single image:

```bash
python predict.py \
  --ckpt runs_custom_final/best.pt \
  --image moth_detector/test/example.jpg \
  --out preds/prediction.png \
  --score_thr 0.7 \
  --device cpu
```

The output image contains predicted bounding boxes and confidence scores.

## Notes

- CPU training is supported but slow; GPU (CUDA) is recommended if available.
- A higher score threshold reduces false positives but may miss detections.
