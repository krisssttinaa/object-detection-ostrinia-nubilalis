import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

_STYLE_KV = re.compile(r"\s*([a-zA-Z\-]+)\s*:\s*([^;]+)\s*;?")

def _strip_unit(x: str) -> float:
    x = (x or "").strip()
    if not x:
        return 0.0
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

def _parse_style(style: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    s = (style or "").strip()
    if not s:
        return out
    for m in _STYLE_KV.finditer(s):
        out[m.group(1).strip().lower()] = m.group(2).strip()
    return out

def _norm_color(c: str) -> str:
    c = (c or "").strip().lower()
    if not c:
        return ""
    if c == "none":
        return ""
    if c.startswith("rgb(") and c.endswith(")"):
        try:
            nums = c[4:-1].split(",")
            r = int(float(nums[0]))
            g = int(float(nums[1]))
            b = int(float(nums[2]))
            return "#{:02x}{:02x}{:02x}".format(r, g, b)
        except Exception:
            return c
    return c

def _local_tag(el: ET.Element) -> str:
    return el.tag.split("}")[-1].lower().strip()

def _parse_viewbox(vb: str) -> Tuple[float, float, float, float]:
    parts = (vb or "").replace(",", " ").split()
    if len(parts) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)

def _rect_xywh(rect: ET.Element) -> Tuple[float, float, float, float]:
    def f(k: str) -> float:
        v = (rect.attrib.get(k, "0") or "0").strip()
        return _strip_unit(v)
    return f("x"), f("y"), f("width"), f("height")

def parse_svg_green_boxes(svg_path: str, img_w: int, img_h: int) -> np.ndarray:
    """
    returns Nx4 float32 [x1,y1,x2,y2] in image pixel coordinates.
    accepts green boxes defined by:
      - id starts with flyManual OR
      - stroke is green in either attribute stroke="..." OR style="stroke:..."
    skips helper flyArea rectangles
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    vb_attr = root.attrib.get("viewBox") or root.attrib.get("viewbox") or ""
    if vb_attr:
        minx, miny, vbw, vbh = _parse_viewbox(vb_attr)
    else:
        vbw = _strip_unit(root.attrib.get("width", "0"))
        vbh = _strip_unit(root.attrib.get("height", "0"))
        minx, miny = 0.0, 0.0
        #fallback if SVG omits both viewBox and w/h
        if vbw <= 0:
            vbw = float(img_w)
        if vbh <= 0:
            vbh = float(img_h)

    if vbw <= 0 or vbh <= 0:
        return np.zeros((0, 4), dtype=np.float32)

    sx = float(img_w) / float(vbw)
    sy = float(img_h) / float(vbh)
    boxes: List[List[float]] = []
    for el in root.iter():
        if _local_tag(el) != "rect":
            continue
        rid = (el.attrib.get("id") or "").strip()
        visibility = (el.attrib.get("visibility") or "").strip().lower()
        if visibility == "hidden":
            continue
        if rid == "flyArea" or rid.lower().startswith("flyarea"):
            continue

        style = _parse_style(el.attrib.get("style", ""))
        stroke_attr = _norm_color(el.attrib.get("stroke", ""))
        stroke_style = _norm_color(style.get("stroke", ""))
        stroke = stroke_attr or stroke_style
        is_green = stroke in {"#41d100", "#00ff00", "#00ff00ff", "lime", "green"}
        is_manual = rid.lower().startswith("flymanual")

        if not (is_green or is_manual):
            continue
        x, y, w, h = _rect_xywh(el)
        if w <= 1 or h <= 1:
            continue
        x1 = (x - minx) * sx
        y1 = (y - miny) * sy
        x2 = (x + w - minx) * sx
        y2 = (y + h - miny) * sy
        x1 = max(0.0, min(float(img_w), x1))
        y1 = max(0.0, min(float(img_h), y1))
        x2 = max(0.0, min(float(img_w), x2))
        y2 = max(0.0, min(float(img_h), y2))
        if x2 > x1 + 1 and y2 > y1 + 1:
            boxes.append([x1, y1, x2, y2])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)

class MothDetectionDataset(Dataset):
    """
    split_dir containing *.jpg and matching *.svg with same stem
    labels: always 1 for moth (ostrinia), background is implicit (0)
    empty images are valid and important for learning background
    """
    def __init__(self, split_dir: str):
        self.split_dir = split_dir
        self.images = sorted([f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".jpeg"))])
        if not self.images:
            raise FileNotFoundError(f"No JPG images found in {split_dir}")

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
        boxes = parse_svg_green_boxes(svg_path, img_w, img_h)
        labels = np.ones((boxes.shape[0],), dtype=np.int64)
        arr = np.array(img, dtype=np.uint8, copy=True)
        img_t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        if boxes_t.numel() > 0:
            area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        target: Dict[str, Any] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64),}
        return img_t, target

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)