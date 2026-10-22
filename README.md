# European Corn Borer Detection (Ostrinia nubilalis)

Object detection of the European corn borer (*Ostrinia nubilalis*) in trap photographs using a custom Faster R-CNN architecture with a ResNet-50/FPN backbone, focal classification loss, CIoU bounding-box regression, and anchors adapted to the training annotations.

## Custom Faster R-CNN modifications

This project builds on **Faster R-CNN with a ResNet-50 + Feature Pyramid Network (FPN) backbone** and introduces three changes to the detection pipeline:

1. **Focal Loss (ROI classification).** The ROI classifier uses focal loss instead of ordinary cross-entropy to reduce the influence of easy examples and give relatively more weight to hard-to-classify examples. The implementation uses `gamma=2.0`, `alpha_pos=0.25`, and `alpha_neg=0.75` by default. See `focal_loss()` and `CustomRoIHeads` in `model_faster_rcnn.py`.
2. **Complete Intersection over Union (CIoU) Loss (ROI box regression).** The ROI bounding-box regression loss is replaced with CIoU loss, which accounts for intersection-over-union, center-point distance, and aspect-ratio consistency between predicted and target boxes. See `complete_iou_loss()` and `CustomRoIHeads` in `model_faster_rcnn.py`.
3. **Data-adaptive anchors (Region Proposal Network).** Instead of using only fixed default anchor dimensions, `derive_anchors()` in `train_model.py` estimates **five anchor scales and three aspect ratios** from bounding-box annotations in the **training split only**. These settings configure the RPN anchor generator, are stored in training checkpoints, and are restored for evaluation and inference. Using training annotations only avoids leaking validation/test labels into anchor configuration.

**Scope:** Focal Loss and CIoU Loss are applied to the **ROI head**, not to the RPN's own classification and regression losses. The rest of the Faster R-CNN training architecture is preserved.

## Repository layout

| Path | Purpose |
| --- | --- |
| `dataset.py` | Image loading and SVG bounding-box annotation parsing |
| `model_faster_rcnn.py` | Model construction, custom ROI heads and loss functions |
| `train_model.py` | Training and validation-loss checkpointing |
| `evaluate_model.py` | Detection evaluation on annotated images |
| `predict.py` | Inference and rendering of predictions for one image |
| `count_empty_svgs.py` | Count empty versus annotated SVGs |
| `library_model/` | Separate experimental/legacy model implementation; retained for reproducibility |
| `moth_detector/` | Local data root; dataset files are not distributed in this repository |

## Setup

Python 3.10–3.12 is recommended. Install a matching PyTorch/TorchVision build for your hardware, as described in the [official installation guide](https://pytorch.org/get-started/locally/). For a standard CPU environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. Training requires a functional PyTorch/TorchVision installation and sufficient RAM. The initial training run may download ImageNet-pretrained ResNet-50 backbone weights, so internet access is needed for that step. Prediction/evaluation from a checkpoint do **not** require that download.

## Data format

Provide your own images and SVG annotations in corresponding train/validation/test splits:

```text
moth_detector/
  train/
    image_001.jpg
    image_001.svg
  val/
    image_002.jpg
    image_002.svg
  test/
    image_003.jpg
    image_003.svg
```

Each `.jpg` or `.jpeg` must have an `.svg` annotation with the same base name. The loader interprets green target bounding boxes (including `flyManual` rectangles); see `dataset.py` for the exact SVG conventions. Images without targets are supported when the SVG contains no target rectangles. The dataset and trained checkpoint files are **not included** in the published source archive.

## Train

```bash
python train_model.py --data_root moth_detector --out_dir runs_focal_ciou --epochs 10 --batch_size 1 --device cpu
```

For a suitable NVIDIA GPU, use `--device cuda`; for supported Apple Silicon setups, `--device mps`. Training saves `runs_focal_ciou/best.pt` (lowest validation loss) and `runs_focal_ciou/last.pt`. Checkpoints include the fitted anchor sizes and ratios.

## Evaluate

```bash
python evaluate_model.py --test_dir moth_detector/test --ckpt runs_focal_ciou/best.pt --device cpu --iou_thr 0.5 --score_thr 0.7
```

The evaluation script reports detection precision, recall, F1, and AP at the specified IoU threshold. A trained checkpoint and a test set are required.

## Predict

```bash
python predict.py --ckpt runs_focal_ciou/best.pt --image moth_detector/test/image_003.jpg --out preds/prediction.png --score_thr 0.7 --device cpu
```

This writes an annotated image. The actual input filename must exist in your local dataset. Checkpoint-specific anchor settings are restored automatically.

## Annotation statistics

```bash
python count_empty_svgs.py
```

The statistics script uses the `moth_detector/{train,val,test}` directories by default.

## Reproducibility and limitations

- The code runs on CPU, though full training will be slow. CUDA/MPS availability depends on your local PyTorch install.
- Model checkpoints, images, and labels are external assets and must be supplied separately. There are no trained weights bundled with this repository.
- The scripts are research-oriented, not a production inference API.
- The separate `library_model/` implementation is preserved as originally supplied; the commands above use the top-level scripts.
- PyTorch and TorchVision are tightly coupled: select compatible versions; the bounds in `requirements.txt` are not a guarantee that every combination works.

## License

See [LICENSE](LICENSE).
