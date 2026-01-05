import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
from model import build_model

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out", type=str, default="pred.png")
    ap.add_argument("--score_thr", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=2).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = Image.open(args.image).convert("RGB")
    img_w, img_h = img.size
    img_t = torch.from_numpy(np.array(img, dtype=np.uint8)) \
        .permute(2, 0, 1).float() / 255.0
    outputs = model([img_t.to(device)])
    pred = outputs[0]

    boxes = pred["boxes"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()

    #FILTERING
    MAX_AREA_RATIO = 0.02    #2% of image, big box dies here
    MIN_AREA_RATIO = 0.0002  #tiny noise removal

    #debug: show largest proposed box
    if len(boxes) > 0:
        areas = []
        for b in boxes:
            x1, y1, x2, y2 = [float(v) for v in b]
            areas.append(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))
        max_ratio = max(areas) / (img_w * img_h)
        print(f"[DEBUG] largest_box_area_ratio = {max_ratio:.6f}")

    filtered = []
    for b, s in zip(boxes, scores):
        s = float(s)
        if s < args.score_thr:
            continue

        x1, y1, x2, y2 = [float(v) for v in b]

        #clamp to image
        x1 = max(0.0, min(x1, img_w - 1))
        y1 = max(0.0, min(y1, img_h - 1))
        x2 = max(0.0, min(x2, img_w - 1))
        y2 = max(0.0, min(y2, img_h - 1))
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        area_ratio = (w * h) / (img_w * img_h)

        if area_ratio > MAX_AREA_RATIO:
            continue
        if area_ratio < MIN_AREA_RATIO:
            continue

        filtered.append((x1, y1, x2, y2, s))
    print(f"[DEBUG] kept {len(filtered)} boxes after filtering")

    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2, s in filtered:
        draw.rectangle([x1, y1, x2, y2], outline="white", width=3)
        draw.text((x1 + 2, y1 + 2), f"{s:.2f}", fill="white")
    img.save(args.out)
    print(f"Saved {args.out} with {len(filtered)} detections (score_thr={args.score_thr})")


if __name__ == "__main__": main()