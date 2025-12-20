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
    viewbox: Tuple[float, float, float, float]  # (minx, miny, vbw, vbh)


def _strip_unit(s: str) -> float:
    # Handles "123", "123px", "123.4pt", etc.
    s = s.strip()
    num = ""
    for ch in s:
        if ch.isdigit() or ch in ".-+eE":
            num += ch
        else:
            break
    return float(num) if num else 0.0


def _parse_viewbox(vb: str) -> Tuple[float, float, float, float]:
    parts = vb.replace(",", " ").split()
    if len(parts) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(float(x) for x in parts)  # type: ignore


def _get_svg_meta(root: ET.Element) -> SvgMeta:
    w = root.attrib.get("width", "0")
    h = root.attrib.get("height", "0")
    svg_w = _strip_unit(w)
    svg_h = _strip_unit(h)

    vb_attr = root.attrib.get("viewBox") or root.attrib.get("viewbox") or ""
    if vb_attr:
        vb = _parse_viewbox(vb_attr)
    else:
        # If no viewBox, assume 0,0,width,height in SVG coordinate space
        vb = (0.0, 0.0, svg_w, svg_h)

    # If width/height missing in SVG, fall back to viewBox size
    if svg_w <= 0 and vb[2] > 0:
        svg_w = vb[2]
    if svg_h <= 0 and vb[3] > 0:
        svg_h = vb[3]

    return SvgMeta(svg_width=svg_w, svg_height=svg_h, viewbox=vb)


def _iter_rects(root: ET.Element) -> List[ET.Element]:
    rects = []
    # SVG namespace handling
    for el in root.iter():
        tag = el.tag.split("}")[-1]  # strip namespace
        if tag == "rect":
            rects.append(el)
    return rects


def parse_svg_to_boxes(svg_path: str, img_w: int, img_h: int) -> np.ndarray:
    """
    Parses an SVG containing rectangles and returns Nx4 boxes in image pixel coords:
    [x1, y1, x2, y2] with 0 <= x1 < x2 <= img_w, 0 <= y1 < y2 <= img_h

    This assumes the SVG's rectangles are the annotations (like your green boxes).
    Supports viewBox scaling.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    meta = _get_svg_meta(root)
    minx, miny, vbw, vbh = meta.viewbox

    # Scale SVG coords -> image pixels
    # SVG user units are mapped via viewBox to width/height.
    # Then we map to image size.
    if vbw <= 0 or vbh <= 0:
        # fallback: treat svg width/height as coordinate space
        vbw = meta.svg_width if meta.svg_width > 0 else img_w
        vbh = meta.svg_height if meta.svg_height > 0 else img_h
        minx, miny = 0.0, 0.0

    sx = img_w / vbw
    sy = img_h / vbh

    boxes: List[List[float]] = []
    for r in _iter_rects(root):
        x = _strip_unit(r.attrib.get("x", "0"))
        y = _strip_unit(r.attrib.get("y", "0"))
        w = _strip_unit(r.attrib.get("width", "0"))
        h = _strip_unit(r.attrib.get("height", "0"))

        # Handle transforms minimally (common case: none). If you later see transforms,
        # we’ll extend this, but for green-box label SVGs it's usually not needed.
        x1 = (x - minx) * sx
        y1 = (y - miny) * sy
        x2 = (x + w - minx) * sx
        y2 = (y + h - miny) * sy

        # Clamp
        x1 = max(0.0, min(float(img_w - 1), x1))
        y1 = max(0.0, min(float(img_h - 1), y1))
        x2 = max(0.0, min(float(img_w), x2))
        y2 = max(0.0, min(float(img_h), y2))

        # Discard degenerate
        if x2 > x1 + 1 and y2 > y1 + 1:
            boxes.append([x1, y1, x2, y2])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)

    return np.array(boxes, dtype=np.float32)


class MothDetectionDataset(Dataset):
    """
    Expects a directory with *.jpg and matching *.svg files (same stem).
    Returns torchvision-style targets for Faster R-CNN.
    """
    def __init__(self, split_dir: str):
        self.split_dir = split_dir
        self.images = []
        for fn in os.listdir(split_dir):
            if fn.lower().endswith(".jpg") or fn.lower().endswith(".jpeg"):
                self.images.append(fn)
        self.images.sort()

        # verify pairs exist
        missing = []
        for img_fn in self.images:
            stem = os.path.splitext(img_fn)[0]
            svg_path = os.path.join(split_dir, stem + ".svg")
            if not os.path.exists(svg_path):
                missing.append(img_fn)
        if missing:
            raise RuntimeError(
                "Missing SVG labels for images:\n" + "\n".join(missing[:50])
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_fn = self.images[idx]
        stem = os.path.splitext(img_fn)[0]
        img_path = os.path.join(self.split_dir, img_fn)
        svg_path = os.path.join(self.split_dir, stem + ".svg")

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        boxes = parse_svg_to_boxes(svg_path, img_w, img_h)
        num = boxes.shape[0]

        # One class: moth = 1 (background is implicit)
        labels = np.ones((num,), dtype=np.int64)

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