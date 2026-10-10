"""猫狗数据集准备：下载 / 解压 / 统计 / 划分。

数据集：Kaggle *Dogs vs. Cats*。这里使用其公开镜像
`kagglecatsanddogs_5340.zip`（Microsoft 托管），内容与 Kaggle 版一致：
`PetImages/Cat` 与 `PetImages/Dog` 各 12500 张。

数据保存在**仓库之外**的 `../data/catsdogs`，避免把大文件提交进 git。

用法：
    python data_prep.py --all            # 下载 + 解压 + 统计 + 划分
    python data_prep.py --download
    python data_prep.py --count
    python data_prep.py --split
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- 目录约定
HERE = Path(__file__).resolve().parent           # .../Jotang-ml-task2/cats-vs-dogs
DATA_ROOT = HERE.parent.parent / "data" / "catsdogs"
ZIP_PATH = DATA_ROOT / "kagglecatsanddogs_5340.zip"
PET_DIR = DATA_ROOT / "PetImages"
SPLIT_DIR = DATA_ROOT / "splits"
MANIFEST = DATA_ROOT / "manifest.csv"

ZIP_URL = (
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/"
    "kagglecatsanddogs_5340.zip"
)
CLASSES = ["Cat", "Dog"]
LABELS = {"Cat": 0, "Dog": 1}

# 划分规模（CPU 训练，先用子集；可按需调大）
SPLIT_SIZES = {"train": 8000, "val": 1000, "test": 2000}
SEED = 42


# ---------------------------------------------------------------- 下载 / 解压
def download(force: bool = False) -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and not force:
        print(f"[skip] 压缩包已存在: {ZIP_PATH} "
              f"({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")
        return ZIP_PATH

    print(f"[download] {ZIP_URL}")
    t0 = time.time()
    last = {"t": 0.0, "n": 0}

    def _hook(block_num, block_size, total_size):
        got = block_num * block_size
        now = time.time()
        if now - last["t"] > 3 or got >= total_size:
            last["t"] = now
            pct = 100 * got / total_size if total_size > 0 else 0
            print(f"    {got / 1e6:8.1f} MB  ({pct:5.1f}%)", flush=True)

    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH, reporthook=_hook)
    print(f"[download] 完成 {ZIP_PATH.stat().st_size / 1e6:.1f} MB，"
          f"耗时 {time.time() - t0:.0f}s")
    return ZIP_PATH


def extract(force: bool = False) -> Path:
    if PET_DIR.exists() and not force and any(PET_DIR.rglob("*.jpg")):
        print(f"[skip] 已解压: {PET_DIR}")
        return PET_DIR
    print(f"[extract] -> {PET_DIR}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_ROOT)
    print("[extract] 完成")
    return PET_DIR


# ---------------------------------------------------------------- 统计 / 清单
def _laplacian_var(gray: np.ndarray) -> float:
    """4 邻域拉普拉斯响应的方差，作为“清晰度”的粗略指标（numpy 实现）。"""
    g = gray.astype(np.float64)
    lap = (
        -4 * g[1:-1, 1:-1]
        + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    )
    return float(lap.var())


def build_manifest(progress_every: int = 2000) -> list[dict]:
    """遍历全部图片，验证可读性并记录尺寸/清晰度，写出 manifest.csv。"""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    broken: list[str] = []

    for cls in CLASSES:
        files = sorted((PET_DIR / cls).glob("*.jpg"))
        print(f"[scan] {cls}: 发现 {len(files)} 个文件")
        for i, fp in enumerate(files, 1):
            try:
                im = Image.open(fp)
                im.load()
                im = im.convert("RGB")
                w, h = im.size
                small = im.resize((64, 64))
                gray = np.asarray(small.convert("L"))
                sharp = _laplacian_var(gray)
            except Exception as exc:  # 损坏文件
                broken.append(f"{fp}\t{type(exc).__name__}: {exc}")
                continue
            rows.append({
                "path": str(fp.relative_to(DATA_ROOT)).replace("\\", "/"),
                "class": cls,
                "label": LABELS[cls],
                "width": w,
                "height": h,
                "sharpness": round(sharp, 2),
            })
            if i % progress_every == 0:
                print(f"    已扫描 {i}/{len(files)} ...", flush=True)

    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class", "label",
                                               "width", "height", "sharpness"])
        writer.writeheader()
        writer.writerows(rows)

    (DATA_ROOT / "broken_files.txt").write_text(
        "\n".join(broken) + ("\n" if broken else ""), encoding="utf-8")
    print(f"[scan] 有效图片 {len(rows)} 张，损坏/跳过 {len(broken)} 张 -> {MANIFEST}")
    return rows


def count_by_class(rows: list[dict]) -> dict:
    counts = {c: 0 for c in CLASSES}
    for r in rows:
        counts[r["class"]] += 1
    return counts


# ---------------------------------------------------------------- 划分
def make_splits(rows: list[dict]) -> dict:
    """分层随机划分为 train/val/test，写出 CSV。"""
    rng = random.Random(SEED)
    by_class: dict[str, list[dict]] = {c: [] for c in CLASSES}
    for r in rows:
        by_class[r["class"]].append(r)

    per_class_train = SPLIT_SIZES["train"] // len(CLASSES)
    per_class_val = SPLIT_SIZES["val"] // len(CLASSES)
    per_class_test = SPLIT_SIZES["test"] // len(CLASSES)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for cls in CLASSES:
        items = by_class[cls][:]
        rng.shuffle(items)
        need = per_class_train + per_class_val + per_class_test
        if len(items) < need:
            raise ValueError(f"{cls} 数量不足：{len(items)} < {need}")
        splits["train"] += items[:per_class_train]
        splits["val"] += items[per_class_train:per_class_train + per_class_val]
        splits["test"] += items[per_class_train + per_class_val:need]

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        rng.shuffle(items)
        with (SPLIT_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "class", "label",
                                                   "width", "height", "sharpness"])
            writer.writeheader()
            writer.writerows(items)
        c = count_by_class(items)
        print(f"[split] {name:5s} -> {len(items):5d} 张  (Cat {c['Cat']} / Dog {c['Dog']})")
    return splits


def load_split(name: str) -> list[dict]:
    with (SPLIT_DIR / f"{name}.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_manifest() -> list[dict]:
    with MANIFEST.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="猫狗数据集准备")
    ap.add_argument("--all", action="store_true", help="下载+解压+统计+划分")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--force", action="store_true", help="强制重新下载/解压")
    args = ap.parse_args(argv)

    if not any([args.all, args.download, args.count, args.split]):
        args.all = True

    if args.all or args.download:
        download(force=args.force)
        extract(force=args.force)
    if args.all or args.count:
        rows = build_manifest()
        counts = count_by_class(rows)
        print(f"[count] 全部有效图片: 总 {len(rows)}  "
              f"Cat {counts['Cat']}  Dog {counts['Dog']}")
    if args.all or args.split:
        if not MANIFEST.exists():
            rows = build_manifest()
        else:
            rows = load_manifest()
        make_splits(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
