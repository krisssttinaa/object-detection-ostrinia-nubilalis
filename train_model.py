import argparse
import os
import time
from typing import Dict
import torch
from torch.utils.data import DataLoader
from dataset import MothDetectionDataset, collate_fn
from model_faster_rcnn import build_custom_model

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_val_loss(model, loader, device) -> float:
    #torchvision detection models only return losses when model.training == True
    #so for validation loss, we temporarily set train() but run under no-grad,
    #and restore the original mode afterwards
    was_training = model.training
    model.train()  #forces loss dict output
    total = 0.0
    n = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            total += float(loss.item())
            n += 1
    #restore original mode
    if not was_training:
        model.eval()
    return total / max(1, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs_custom")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight_decay", type=float, default=0.0005)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    args = ap.parse_args()
    set_seed(args.seed)
    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")
    train_ds = MothDetectionDataset(train_dir)
    val_ds = MothDetectionDataset(val_dir)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn)

    if args.device == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    model = build_custom_model(num_classes=2).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    #torchvision reference-style hyperparams
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    use_amp = (device.type == "cuda")
    scaler = None
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        steps = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            running += float(loss.item())
            steps += 1
            """if steps % 5 == 0:
                print(f"  step {steps} | loss={float(loss.item()):.4f}", flush=True)"""

        lr_sched.step()
        train_loss = running / max(1, steps)
        val_loss = evaluate_val_loss(model, val_loader, device)
        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={dt:.1f}s")
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, last_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, best_path)
            print(f"  -> saved new best: {best_path} (val_loss={best_val:.4f})")
    print("Training finished.")
    print("Best:", best_path)
    print("Last:", last_path)

if __name__ == "__main__": main()