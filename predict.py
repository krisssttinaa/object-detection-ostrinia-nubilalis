import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
from model_faster_rcnn import build_custom_model

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out", type=str, default="pred_custom.png")
    ap.add_argument("--score_thr", type=float, default=0.7)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
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
    img = Image.open(args.image).convert("RGB")
    img_t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
    det = model([img_t.to(device)])[0]
    boxes = det["boxes"].detach().cpu().numpy()
    scores = det["scores"].detach().cpu().numpy()
    draw = ImageDraw.Draw(img)
    kept = 0
    for b, s in zip(boxes, scores):
        if float(s) < args.score_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in b]
        draw.rectangle([x1, y1, x2, y2], outline="white", width=3)
        draw.text((x1 + 2, y1 + 2), f"{float(s):.2f}", fill="white")
        kept += 1
    img.save(args.out)
    print(f"Saved {args.out} | kept={kept} (score_thr={args.score_thr})")

if __name__ == "__main__": main()