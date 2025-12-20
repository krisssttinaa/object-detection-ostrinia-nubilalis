import argparse
import os
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset import MothDetectionDataset, collate_fn
from model import build_model


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_val_loss(model, loader, device) -> float:
    model.train()  # IMPORTANT: detection models return losses in train mode
    total = 0.0
    n = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict: Dict[str, torch.Tensor] = model(images, targets)
        loss = sum(loss_dict.values()).item()
        total += loss
        n += 1
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Path to DATASET_ROOT containing train/val/test folders")
    ap.add_argument("--out_dir", type=str, default="runs_moth", help="Where to save checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    train_ds = MothDetectionDataset(train_dir)
    val_ds = MothDetectionDataset(val_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #if torch.backends.mps.is_available():
    #    device = torch.device("mps")
    #elif torch.cuda.is_available():
    #    device = torch.device("cuda")
    #else:
    #    device = torch.device("cpu")

    print("Using device:", device)
    
    model = build_model(num_classes=2).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.3)

    os.makedirs(args.out_dir, exist_ok=True)

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        steps = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running += loss.item()
            steps += 1

        lr_sched.step()

        train_loss = running / max(1, steps)
        val_loss = evaluate_val_loss(model, val_loader, device)
        dt = time.time() - t0

        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={dt:.1f}s")

        # Save last
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, last_path)

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, best_path)
            print(f"  -> saved new best: {best_path} (val_loss={best_val:.4f})")

    print("Training finished.")
    print(f"Best checkpoint: {best_path} (val_loss={best_val:.4f})")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()