import argparse
import os
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

    img = Image.open(args.image).convert("RGB")
    img_t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
    out = model([img_t.to(device)])[0]

    boxes = out["boxes"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()

    draw = ImageDraw.Draw(img)
    kept = 0
    for b, s in zip(boxes, scores):
        if float(s) < args.score_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in b]
        draw.rectangle([x1, y1, x2, y2], width=3)
        draw.text((x1, y1), f"{s:.2f}")
        kept += 1

    img.save(args.out)
    print(f"Saved {args.out} with {kept} detections (score_thr={args.score_thr}).")


if __name__ == "__main__":
    main()
