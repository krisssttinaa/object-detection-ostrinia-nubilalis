import os
import argparse
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import numpy as np
import torch
from PIL import Image
from model_faster_rcnn import build_custom_model

SVG_NS = "http://www.w3.org/2000/svg"
LABEL_STROKE = "#41D100"
LABEL_ID_PREFIX = "flyManual"

def iter_images(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png"}
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(r, f))
    return sorted(out)

def image_to_svg_path(img_path: str) -> str:
    base, _ = os.path.splitext(img_path)
    return base + ".svg"

def parse_svg_boxes(svg_path: str) -> List[List[float]]:
    """list of GT boxes [x1,y1,x2,y2] for Ostrinia"""
    if not os.path.exists(svg_path):
        return []
    try:
        root = ET.parse(svg_path).getroot()
    except Exception:
        return []

    boxes = []
    def iter_rects():
        for el in root.iter():
            if el.tag.endswith("rect"):
                yield el

    for r in iter_rects():
        rid = (r.get("id") or "").strip()
        stroke = (r.get("stroke") or "").strip().upper()
        is_label = rid.startswith(LABEL_ID_PREFIX) or stroke == LABEL_STROKE
        if not is_label:
            continue
        #SVG rect: x,y,w,h
        x = float(r.get("x", "0"))
        y = float(r.get("y", "0"))
        w = float(r.get("width", "0"))
        h = float(r.get("height", "0"))
        x1, y1, x2, y2 = x, y, x + w, y + h
        if w > 0 and h > 0:
            boxes.append([x1, y1, x2, y2])
    return boxes

def iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """AP computation using precision envelope"""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--test_dir", required=True, type=str)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--score_thr", default=0.7, type=float)
    ap.add_argument("--iou_thr", default=0.5, type=float)
    ap.add_argument("--max_images", default=10_000, type=int)
    args = ap.parse_args()

    if args.device == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model = build_custom_model(num_classes=2).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    img_paths = iter_images(args.test_dir)[: args.max_images]
    #P/R/F1 at fixed threshold
    TP = 0
    FP = 0
    FN = 0
    #AP: gather all predictions across dataset
    all_scores = []
    all_matches = []  #1 if TP, 0 if FP, after greedy matching per-image
    total_gt = 0

    for img_path in img_paths:
        svg_path = image_to_svg_path(img_path)
        gt_boxes = parse_svg_boxes(svg_path)
        total_gt += len(gt_boxes)
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        img_t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        det = model([img_t.to(device)])[0]
        pred_boxes = det["boxes"].detach().cpu().numpy().tolist()
        pred_scores = det["scores"].detach().cpu().numpy().tolist()

        #filter by score threshold for P/R/F1 counts
        kept = [(b, s) for b, s in zip(pred_boxes, pred_scores) if float(s) >= args.score_thr]
        #greedy matching: each GT can match at most 1 prediction
        matched_gt = [False] * len(gt_boxes)
        img_TP = 0
        img_FP = 0
        #for AP, we want predictions (not thresholded), but still matched per-image in score order
        order = np.argsort(-np.array(pred_scores))
        pred_boxes_sorted = [pred_boxes[i] for i in order]
        pred_scores_sorted = [pred_scores[i] for i in order]

        matched_gt_ap = [False] * len(gt_boxes)
        for b, s in zip(pred_boxes_sorted, pred_scores_sorted):
            best_iou = 0.0
            best_j = -1
            for j, gtb in enumerate(gt_boxes):
                if matched_gt_ap[j]:
                    continue
                v = iou(b, gtb)
                if v > best_iou:
                    best_iou = v
                    best_j = j
            if best_iou >= args.iou_thr and best_j >= 0:
                matched_gt_ap[best_j] = True
                all_matches.append(1)
            else:
                all_matches.append(0)
            all_scores.append(float(s))

        #TP/FP/FN at fixed threshold
        for b, s in kept:
            best_iou = 0.0
            best_j = -1
            for j, gtb in enumerate(gt_boxes):
                if matched_gt[j]:
                    continue
                v = iou(b, gtb)
                if v > best_iou:
                    best_iou = v
                    best_j = j
            if best_iou >= args.iou_thr and best_j >= 0:
                matched_gt[best_j] = True
                img_TP += 1
            else:
                img_FP += 1

        img_FN = matched_gt.count(False)
        TP += img_TP
        FP += img_FP
        FN += img_FN

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    #APIoU threshold
    if total_gt == 0:
        ap50 = 0.0
    else:
        #sort all predictions globally by score
        idx = np.argsort(-np.array(all_scores))
        matches_sorted = np.array(all_matches)[idx]
        tp_cum = np.cumsum(matches_sorted)
        fp_cum = np.cumsum(1 - matches_sorted)
        recalls = tp_cum / float(total_gt)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap50 = average_precision(recalls, precisions)

    print("Evaluation on TEST set")
    print(f"Images evaluated: {len(img_paths)}")
    print(f"IoU threshold: {args.iou_thr}")
    print(f"Score threshold (P/R/F1): {args.score_thr}")
    print()
    print(f"TP={TP} | FP={FP} | FN={FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AP{args.iou_thr:.1f}: {ap50:.4f}")

if __name__ == "__main__": main()