import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

@dataclass
class SvgMeta:
    svg_width: float
    svg_height: float
    viewbox: Tuple[float, float, float, float]  #(minx, miny, vbw, vbh)

def _strip_unit(x: str) -> float:
    x = (x or "").strip()
    if not x:
        return 0.0
    #handle percentages (not meaningful for root width/height)
    if x.endswith("%"):
        return 0.0
    for suf in ["px", "pt", "cm", "mm", "in"]:
        if x.endswith(suf):
            x = x[:-len(suf)]
            break
    try:
        return float(x)
    except Exception:
        return 0.0

def _parse_viewbox(vb: str) -> Tuple[float, float, float, float]:
    parts = (vb or "").replace(",", " ").split()
    if len(parts) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(float(x) for x in parts)

def _get_svg_meta(root: ET.Element) -> SvgMeta:
    w = root.attrib.get("width", "0")
    h = root.attrib.get("height", "0")
    svg_w = _strip_unit(w)
    svg_h = _strip_unit(h)
    vb_attr = root.attrib.get("viewBox") or root.attrib.get("viewbox") or ""
    if vb_attr:
        vb = _parse_viewbox(vb_attr)
    else:
        vb = (0.0, 0.0, svg_w, svg_h)

    #if width/height missing, use viewBox size
    if svg_w <= 0 and vb[2] > 0:
        svg_w = vb[2]
    if svg_h <= 0 and vb[3] > 0:
        svg_h = vb[3]
    return SvgMeta(svg_width=svg_w, svg_height=svg_h, viewbox=vb)

def _iter_rects(root: ET.Element) -> List[ET.Element]:
    rects: List[ET.Element] = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]  # strip namespace
        if tag == "rect":
            rects.append(el)
    return rects

def _rect_to_xywh(rect: ET.Element) -> Tuple[float, float, float, float]:
    #SVG rect can have x/y/width/height
    x = float(rect.attrib.get("x", "0") or "0")
    y = float(rect.attrib.get("y", "0") or "0")
    w = float(rect.attrib.get("width", "0") or "0")
    h = float(rect.attrib.get("height", "0") or "0")
    return x, y, w, h

def parse_svg_to_boxes_labels(svg_path: str, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    boxes: Nx4 float32 in image pixel coords [x1,y1,x2,y2]
    labels: N int64 (1=ostrinia, 2=non_target)
    only keeps true manual annotation rectangles, NOT helper rectangles like flyArea.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    meta = _get_svg_meta(root)
    minx, miny, vbw, vbh = meta.viewbox

    #protect against bad viewBox
    if vbw <= 0 or vbh <= 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    sx = float(img_w) / float(vbw)
    sy = float(img_h) / float(vbh)
    boxes: List[List[float]] = []
    labels: List[int] = []
    for rect in _iter_rects(root):
        rid = (rect.attrib.get("id") or "").strip()
        stroke = (rect.attrib.get("stroke") or "").strip().lower()
        visibility = (rect.attrib.get("visibility") or "").strip().lower()
        #1) drop hidden helper rects
        if visibility == "hidden":
            continue
        #2) drop flyArea (huge frame) and other non-manual helper ids
        if rid == "flyArea" or rid.startswith("flyArea"):
            continue
        #3) determine class:
        #    - manual ostrinia boxes are green in your dataset (#41D100) and/or id starts with flyManual
        #    - optional: non_target can be marked red (#FF0000) later
        label = None
        if rid.startswith("flyManual") or stroke in {"#41d100", "#00ff00", "#00ff00ff"}:
            label = 1  # ostrinia
        elif rid.startswith("nonTarget") or stroke in {"#ff0000", "#ff2d2d", "#d10000"}:
            label = 2  # non_target (optional future)
        else:
            #ignore any other rects (borders, guides, etc.)
            continue

        x, y, w, h = _rect_to_xywh(rect)
        if w <= 1 or h <= 1:
            continue

        #convert from SVG user space to viewBox space (account for minx/miny), then to image pixels
        x1 = (x - minx) * sx
        y1 = (y - miny) * sy
        x2 = (x + w - minx) * sx
        y2 = (y + h - miny) * sy

        #clamp
        x1 = max(0.0, min(float(img_w), x1))
        y1 = max(0.0, min(float(img_h), y1))
        x2 = max(0.0, min(float(img_w), x2))
        y2 = max(0.0, min(float(img_h), y2))
        if x2 > x1 + 1 and y2 > y1 + 1:
            boxes.append([x1, y1, x2, y2])
            labels.append(int(label))
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

class MothDetectionDataset(Dataset):
    """Expects split_dir containing *.jpg and matching *.svg with same stem, produces targets for Faster R-CNN"""
    def __init__(self, split_dir: str):
        self.split_dir = split_dir
        self.images: List[str] = []
        for fn in os.listdir(split_dir):
            if fn.lower().endswith(".jpg") or fn.lower().endswith(".jpeg"):
                self.images.append(fn)
        self.images.sort()

        missing = []
        for img_fn in self.images:
            stem = os.path.splitext(img_fn)[0]
            svg_path = os.path.join(split_dir, stem + ".svg")
            if not os.path.exists(svg_path):
                missing.append(svg_path)
        if missing:
            raise FileNotFoundError(f"Missing SVG for {len(missing)} images. Example: {missing[0]}")

    def __len__(self): return len(self.images)

    def __getitem__(self, idx: int):
        img_fn = self.images[idx]
        stem = os.path.splitext(img_fn)[0]
        img_path = os.path.join(self.split_dir, img_fn)
        svg_path = os.path.join(self.split_dir, stem + ".svg")
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        boxes, labels = parse_svg_to_boxes_labels(svg_path, img_w, img_h)

        num = int(boxes.shape[0])
        target: Dict[str, Any] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.as_tensor(
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                dtype=torch.float32
            ) if num > 0 else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((num,), dtype=torch.int64),
        }
        img_t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        return img_t, target

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)