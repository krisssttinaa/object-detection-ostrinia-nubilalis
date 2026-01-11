import os
import xml.etree.ElementTree as ET
from typing import List, Dict

DATASET_ROOT = "moth_detector"
SPLITS = ["train", "val", "test"]
SVG_NS = "http://www.w3.org/2000/svg"
AREA_STROKE = "#00F0FF"
LABEL_STROKE = "#41D100"

def find_svg_files(split_dir: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(split_dir):
        for f in files:
            if f.lower().endswith(".svg"):
                out.append(os.path.join(root, f))
    return out

def parse_svg(svg_path: str) -> ET.Element: return ET.parse(svg_path).getroot()

def iter_rects(root: ET.Element):
    for el in root.iter():
        if el.tag.endswith("rect"):
            yield el

def is_label_rect(rect: ET.Element) -> bool:
    rid = (rect.get("id") or "").strip()
    stroke = (rect.get("stroke") or "").strip()
    if rid.startswith("flyManual"):
        return True
    if stroke.upper() == LABEL_STROKE:
        return True
    return False

def count_labels(svg_path: str) -> int:
    try:
        root = parse_svg(svg_path)
    except Exception as e:
        print(f"[WARN] Could not parse SVG (treated as empty): {svg_path} | {e}")
        return 0
    n = 0
    for r in iter_rects(root):
        if is_label_rect(r):
            n += 1
    return n

def analyze_split(split_name: str) -> Dict:
    split_dir = os.path.join(DATASET_ROOT, split_name)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split folder not found: {split_dir}")
    svg_files = find_svg_files(split_dir)
    total = 0
    empty = 0
    non_empty = 0
    total_labels = 0
    for svg_path in svg_files:
        total += 1
        labels = count_labels(svg_path)
        total_labels += labels
        if labels == 0:
            empty += 1
        else:
            non_empty += 1
    empty_ratio = (empty / total * 100.0) if total > 0 else 0.0
    return {
        "split": split_name,
        "total": total,
        "empty": empty,
        "non_empty": non_empty,
        "empty_ratio": empty_ratio,
        "total_labels": total_labels,}

def main():
    print("Empty-image statistics (SVG contains flyManual_* / #41D100 label boxes)\n")
    grand = {"total": 0, "empty": 0, "non_empty": 0, "total_labels": 0}
    for split in SPLITS:
        s = analyze_split(split)
        grand["total"] += s["total"]
        grand["empty"] += s["empty"]
        grand["non_empty"] += s["non_empty"]
        grand["total_labels"] += s["total_labels"]
        print(f"{split.upper()}:")
        print(f"  Total images (SVGs)   : {s['total']}")
        print(f"  Empty images          : {s['empty']}")
        print(f"  Non-empty images      : {s['non_empty']}")
        print(f"  Empty ratio           : {s['empty_ratio']:.2f}%")
        print(f"  Total labeled boxes   : {s['total_labels']}\n")
    grand_ratio = (grand["empty"] / grand["total"] * 100.0) if grand["total"] > 0 else 0.0
    print("OVERALL:")
    print(f"  Total images (SVGs)   : {grand['total']}")
    print(f"  Empty images          : {grand['empty']}")
    print(f"  Non-empty images      : {grand['non_empty']}")
    print(f"  Empty ratio           : {grand_ratio:.2f}%")
    print(f"  Total labeled boxes   : {grand['total_labels']}")

if __name__ == "__main__": main()