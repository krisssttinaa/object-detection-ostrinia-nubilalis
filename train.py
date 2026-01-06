import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from dataset import MothDetectionDataset, collate_fn
from model import build_model

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device(device_arg: str) -> torch.device:
    d = (device_arg or "auto").lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # auto:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    n = 0
    for images, targets in loader:
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)  # dict of losses in train mode
        loss = sum(loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(1, n)

@torch.no_grad()
def eval_one_epoch_loss(model, loader, device):
    """
    TorchVision detection models do NOT return losses in eval() mode.
    To compute validation loss, temporarily switch to train() while keeping no_grad().
    """
    was_training = model.training
    model.train()  # important: enables loss dict output

    total = 0.0
    n = 0
    for images, targets in loader:
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # dict (because train mode)
        loss = sum(loss_dict.values())
        total += float(loss.item())
        n += 1
    # restore original mode
    if not was_training:
        model.eval()
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Folder containing train/ val/ test/")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="runs_moth")
    ap.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--num_classes", type=int, default=2, help="includes background. 2=bg+ostrinia, 3=bg+ostrinia+non_target")
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
        collate_fn=collate_fn)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn)

    device = pick_device(args.device)
    print("Using device:", device)
    print("Train images:", len(train_ds), "| Val images:", len(val_ds))
    print("num_classes:", args.num_classes)

    model = build_model(num_classes=args.num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.3)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = eval_one_epoch_loss(model, val_loader, device)

        lr_sched.step()
        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | time={dt:.1f}s")

        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "num_classes": args.num_classes},
            last_path
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "num_classes": args.num_classes},
                best_path)
            print(f"  -> saved new best: {best_path} (val_loss={best_val:.4f})")
    print("Training finished.")
    print(f"Best checkpoint: {best_path} (val_loss={best_val:.4f})")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__": main()