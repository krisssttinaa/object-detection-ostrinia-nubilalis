import argparse
import os
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import MothDetectionDataset, collate_fn
from model import build_model

def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

def match_detections_to_gt(
    det_boxes: np.ndarray,
    det_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float,
    score_thr: float
) -> Tuple[int, int, int]:
    """
    Returns (TP, FP, FN) for one image using greedy matching.
    """
    keep = det_scores >= score_thr
    det_boxes = det_boxes[keep]
    det_scores = det_scores[keep]

    # sort by score high->low
    order = np.argsort(-det_scores)
    det_boxes = det_boxes[order]
    det_scores = det_scores[order]
    gt_used = np.zeros((gt_boxes.shape[0],), dtype=bool)
    tp = 0
    fp = 0
    for db in det_boxes:
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            v = iou(db, gb)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_iou >= iou_thr and best_j >= 0:
            tp += 1
            gt_used[best_j] = True
        else:
            fp += 1
    fn = int((~gt_used).sum())
    return tp, fp, fn

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--score_thr", type=float, default=0.5)
    args = ap.parse_args()

    test_dir = os.path.join(args.data_root, "test")
    ds = MothDetectionDataset(test_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #if torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #elif torch.cuda.is_available():
    #    device = torch.device("cuda")
    #else:
    #    device = torch.device("cpu")

    #print("Using device:", device)
    model = build_model(num_classes=2).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    TP = FP = FN = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        # batch_size=1 typical; handle general batch anyway
        for out, tgt in zip(outputs, targets):
            det_boxes = out["boxes"].detach().cpu().numpy().astype(np.float32)
            det_scores = out["scores"].detach().cpu().numpy().astype(np.float32)
            gt_boxes = tgt["boxes"].detach().cpu().numpy().astype(np.float32)

            tp, fp, fn = match_detections_to_gt(det_boxes, det_scores, gt_boxes, args.iou_thr, args.score_thr)
            TP += tp
            FP += fp
            FN += fn

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("=== Test metrics ===")
    print(f"IoU threshold: {args.iou_thr}")
    print(f"Score threshold: {args.score_thr}")
    print(f"TP={TP} FP={FP} FN={FN}")
    print(f"Precision={precision:.4f}")
    print(f"Recall   ={recall:.4f}")
    print(f"F1       ={f1:.4f}")

if __name__ == "__main__": main()