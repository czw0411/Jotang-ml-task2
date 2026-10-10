"""训练脚本（任务 3、4；同时支撑增强对照与误差分析）。

用法示例：
    python train.py --model cnn --epochs 10 --aug flip,crop --tag cnn_aug
    python train.py --model mlp --epochs 8 --size 64 --tag mlp_base
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dataset as ds  # noqa: E402
import models as M  # noqa: E402
import viz  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_aug(spec: str) -> dict:
    spec = (spec or "").lower().strip()
    parts = set(p for p in spec.replace("+", ",").split(",") if p)
    return {"flip": "flip" in parts, "crop": "crop" in parts, "color": "color" in parts,
            "none": spec in ("", "none")}


def make_loaders(aug: dict, size: int, batch_size: int, num_workers: int):
    train_tf = ds.tf_train_aug(size, flip=aug["flip"], crop=aug["crop"], color=aug["color"])
    eval_tf = ds.tf_resize_crop(size)
    train_loader = ds.build_loader("train", train_tf, batch_size, num_workers=num_workers)
    val_loader = ds.build_loader("val", eval_tf, batch_size, num_workers=num_workers)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, train: bool):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train:
                optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def fit(model, train_loader, val_loader, epochs, lr, tag, size,
        weight_decay=1e-4, verbose=True, meta=None):
    meta = dict(meta or {})
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "epoch_sec": []}
    best = {"val_acc": -1.0, "epoch": -1, "path": None}
    model.to(DEVICE)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, False)
        sched.step()
        dt = time.time() - t0
        for k, v in (("train_loss", tr_loss), ("train_acc", tr_acc),
                     ("val_loss", va_loss), ("val_acc", va_acc), ("epoch_sec", dt)):
            history[k].append(round(v, 4))
        if va_acc > best["val_acc"]:
            best.update(val_acc=va_acc, epoch=ep)
            path = viz.OUT_DIR / f"{tag}_best.pt"
            torch.save({"tag": tag, "state_dict": model.state_dict(),
                        "val_acc": va_acc, "epoch": ep, "size": size, **meta}, path)
            best["path"] = str(path)
        if verbose:
            print(f"[{tag}] epoch {ep:2d}/{epochs}  "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s", flush=True)
    return history, best


def trace_shapes(model, size: int = 128, batch: int = 1):
    """记录输入流经每个叶子层后的输出形状（任务 4）。"""
    model.eval()
    rows = []
    handles = []

    def make_hook(name, mod):
        def hook(_m, _inp, out):
            shape = tuple(out.shape) if hasattr(out, "shape") else None
            rows.append((name, type(mod).__name__, shape))
        return hook

    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0 and name:
            handles.append(mod.register_forward_hook(make_hook(name, mod)))
    with torch.no_grad():
        model(torch.zeros(batch, 3, size, size, device=next(model.parameters()).device))
    for h in handles:
        h.remove()
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--aug", default="flip,crop", help="flip,crop,color 或 none")
    ap.add_argument("--width", type=int, default=32, help="CNN 通道基数")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    set_seed(42)
    aug = parse_aug(args.aug)
    tag = args.tag or f"{args.model}_{'none' if aug['none'] else 'aug'}"
    size = args.size if args.model == "cnn" else min(args.size, 64)

    print(f"device={DEVICE}  model={args.model}  size={size}  aug={aug}  tag={tag}")
    train_loader, val_loader = make_loaders(
        aug if args.model == "cnn" else {"flip": False, "crop": False, "color": False, "none": True},
        size, args.batch_size, args.workers)

    model = M.SmallCNN(width=args.width) if args.model == "cnn" else M.MLP(size=size)
    print(f"参数量: {M.count_parameters(model):,}")

    rows = trace_shapes(model, size=size)
    print("层输出形状：")
    for name, kind, shape in rows:
        print(f"    {name:28s} {kind:18s} {shape}")

    history, best = fit(model, train_loader, val_loader, args.epochs, args.lr, tag, size,
                        meta={"arch": args.model, "width": args.width})
    viz.save_curves(history, viz.OUT_DIR / f"{tag}_curves.png", title=f"{tag} 训练曲线")
    (viz.OUT_DIR / f"{tag}_history.json").write_text(
        json.dumps({"tag": tag, "model": args.model, "size": size, "aug": aug,
                    "params": M.count_parameters(model), "history": history,
                    "best": best, "shapes": [[n, k, list(s) if s else None] for n, k, s in rows]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"最佳 val acc = {best['val_acc']:.4f} (epoch {best['epoch']}) -> {best['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
