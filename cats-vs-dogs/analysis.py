"""任务 4、5、7：层形状、特征图可视化、错误分析。

用法：
    python analysis.py --shapes --ckpt outputs/cnn_aug_best.pt
    python analysis.py --features --ckpt outputs/cnn_aug_best.pt
    python analysis.py --errors --ckpt outputs/cnn_aug_best.pt
    python analysis.py --all --ckpt outputs/cnn_aug_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import data_prep as dp  # noqa: E402
import dataset as ds  # noqa: E402
import models as M  # noqa: E402
import viz  # noqa: E402

DEVICE = torch.device("cpu")
CLASS_NAMES = {0: "Cat", 1: "Dog"}


def load_model(ckpt_path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if ckpt.get("arch", "cnn") == "mlp":
        model = M.MLP(size=ckpt.get("size", 64))
    else:
        model = M.SmallCNN(width=ckpt.get("width", 32))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(DEVICE)
    return model, ckpt


def show_shapes(model, size: int) -> None:
    import train as T
    rows = T.trace_shapes(model, size=size)
    print(f"{'层':32s}{'类型':18s}{'输出形状 (N,C,H,W)'}")
    print("-" * 78)
    for name, kind, shape in rows:
        print(f"{name:32s}{kind:18s}{shape}")


# ------------------------------------------------------------------ 特征图
def visualize_features(ckpt_path, n_images: int = 3, n_channels: int = 16) -> None:
    model, ckpt = load_model(ckpt_path)
    size = ckpt.get("size", 128)
    tf = ds.tf_resize_crop(size)

    rows = dp.load_split("test")
    # 猫、狗各取一些
    picks = [r for r in rows if r["class"] == "Cat"][: (n_images + 1) // 2] + \
            [r for r in rows if r["class"] == "Dog"][: n_images // 2]

    for pick in picks[:n_images]:
        img = Image.open(dp.DATA_ROOT / pick["path"]).convert("RGB")
        x = tf(img).unsqueeze(0)
        acts = {}
        handles = [
            model.block1[2].register_forward_hook(
                lambda m, i, o: acts.__setitem__("first", o.detach())),
            model.block3[2].register_forward_hook(
                lambda m, i, o: acts.__setitem__("last", o.detach())),
        ]
        with torch.no_grad():
            logits = model(x)
        for h in handles:
            h.remove()

        prob = F.softmax(logits, dim=1)[0]
        pred = int(prob.argmax())
        panels = [("输入图片", np.asarray(img.resize((size, size))))]

        # 第一层卷积特征图
        fmap1 = acts["first"][0]                     # (C, H, W)
        panels.append((f"— 第1层卷积 特征图 {tuple(fmap1.shape)} —", np.zeros((4, 4, 3), np.uint8)))
        for c in range(min(n_channels, fmap1.shape[0])):
            m = fmap1[c].numpy()
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            panels.append((f"first ch{c}", m))

        # 最后一层卷积特征图
        fmap3 = acts["last"][0]
        panels.append((f"— 最后一层卷积 特征图 {tuple(fmap3.shape)} —", np.zeros((4, 4, 3), np.uint8)))
        for c in range(min(n_channels, fmap3.shape[0])):
            m = fmap3[c].numpy()
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            panels.append((f"last ch{c}", m))

        name = f"05_featuremaps_{pick['class']}_{Path(pick['path']).stem}.png"
        out = viz.save_grid(
            panels, viz.OUT_DIR / name, ncols=6, figsize=(15, 12),
            suptitle=f"{pick['class']} 预测={CLASS_NAMES[pred]} p={float(prob[pred]):.3f} "
                     f"| 第1层(浅) vs 最后一层(深) 特征图")
        print("已保存:", out)


# ------------------------------------------------------------------ 错误分析
def _predict(model, size, split="test", batch_size=64, tta: bool = False):
    tf = ds.tf_resize_crop(size)
    loader = ds.build_loader(split, tf, batch_size, shuffle=False, num_workers=4)
    probs, preds, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            p = F.softmax(model(x), dim=1)
            if tta:
                p = (p + F.softmax(model(torch.flip(x, dims=[3])), dim=1)) / 2
            probs.append(p.numpy())
            preds.append(p.argmax(1).numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(preds), np.concatenate(labels)


def error_analysis(ckpt_path, n_show: int = 12, batch_size: int = 64) -> None:
    from sklearn.metrics import confusion_matrix, classification_report
    model, ckpt = load_model(ckpt_path)
    size = ckpt.get("size", 128)
    rows = dp.load_split("test")

    probs, preds, labels = _predict(model, size, "test", batch_size, tta=False)
    acc = float((preds == labels).mean())
    print(f"测试集样本: {len(labels)}   准确率(无TTA): {acc:.4f}")
    cm = confusion_matrix(labels, preds)
    print("混淆矩阵 [[TN,FP],[FN,TP]] (行=真实, 列=预测):\n", cm)
    print(classification_report(labels, preds, target_names=["Cat", "Dog"], digits=4))

    probs_t, preds_t, _ = _predict(model, size, "test", batch_size, tta=True)
    acc_t = float((preds_t == labels).mean())
    print(f"准确率(水平翻转 TTA): {acc_t:.4f}   提升 {acc_t - acc:+.4f}")

    fig_path = viz.save_confusion(cm, viz.OUT_DIR / "06_confusion_matrix.png")
    print("已保存:", fig_path)

    wrong = np.where(preds != labels)[0]
    order = np.argsort(-probs[wrong, preds[wrong]])
    wrong = wrong[order][:n_show]
    panels = []
    fixes = 0
    for i in wrong:
        img = Image.open(dp.DATA_ROOT / rows[i]["path"]).convert("RGB")
        img.thumbnail((280, 280))
        conf = float(probs[i, preds[i]])
        tta_pred = int(preds_t[i])
        tag = CLASS_NAMES[preds[i]]
        if tta_pred == labels[i] and tta_pred != preds[i]:
            fixes += 1
            tag += "→TTA修正"
        panels.append((f"真:{CLASS_NAMES[labels[i]]} 预:{tag} p={conf:.2f}", np.asarray(img)))
    out = viz.save_grid(panels, viz.OUT_DIR / "07_misclassified.png", ncols=4,
                        figsize=(14, 3.4 * ((len(panels) + 3) // 4)),
                        suptitle="预测错误的样本（真值 / 预测 / 置信度）")
    print("已保存:", out)
    print(f"展示 {len(wrong)} 个错误样本，其中 {fixes} 个被水平翻转 TTA 修正")


def improve(ckpt_path, batch_size: int = 64) -> None:
    """改进实验：用验证集选一个判别阈值，缓解“偏向预测猫”的问题。"""
    model, ckpt = load_model(ckpt_path)
    size = ckpt.get("size", 128)

    vp, _, vl = _predict(model, size, "val", batch_size)
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.20, 0.81, 0.01):
        pred = (vp[:, 1] >= t).astype(int)
        rec_cat = float(((pred == 0) & (vl == 0)).sum() / max((vl == 0).sum(), 1))
        rec_dog = float(((pred == 1) & (vl == 1)).sum() / max((vl == 1).sum(), 1))
        score = (rec_cat + rec_dog) / 2          # 平衡准确率
        if score > best_score:
            best_t, best_score = float(t), score
    print(f"[improve] 验证集上最佳判别阈值 t*={best_t:.2f}（平衡准确率 {best_score:.4f}）")

    tp, _, tl = _predict(model, size, "test", batch_size)
    acc_default = float(((tp[:, 1] >= 0.5).astype(int) == tl).mean())
    tuned = (tp[:, 1] >= best_t).astype(int)
    acc_tuned = float((tuned == tl).mean())
    rec_cat = float(((tuned == 0) & (tl == 0)).sum() / max((tl == 0).sum(), 1))
    rec_dog = float(((tuned == 1) & (tl == 1)).sum() / max((tl == 1).sum(), 1))
    print(f"[improve] 测试集准确率: 默认阈值 0.5 -> {acc_default:.4f} | "
          f"阈值 t*={best_t:.2f} -> {acc_tuned:.4f}  ({acc_tuned - acc_default:+.4f})")
    print(f"[improve] 阈值调整后 Cat 召回 {rec_cat:.4f} / Dog 召回 {rec_dog:.4f}"
          f"（更平衡）")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(viz.OUT_DIR / "cnn_aug_best.pt"))
    ap.add_argument("--shapes", action="store_true")
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--images", type=int, default=3)
    args = ap.parse_args(argv)

    model, ckpt = load_model(args.ckpt)
    size = ckpt.get("size", 128)
    if args.all or args.shapes:
        show_shapes(model, size)
    if args.all or args.features:
        visualize_features(args.ckpt, n_images=args.images)
    if args.all or args.errors:
        error_analysis(args.ckpt)
        improve(args.ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
