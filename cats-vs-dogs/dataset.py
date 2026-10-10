"""数据读取与预处理流程（任务 2）。

提供三种缩放/裁剪策略，用来对比“直接拉伸”和“等比缩放+裁剪”的差别：
- `tf_stretch`     : Resize((S,S))，直接拉伸到正方形 —— 宽高比被改变，可能变形；
- `tf_resize_crop` : 先按短边等比缩放到 S*1.14，再中心裁剪 —— 保持比例，不拉伸；
- `tf_train_aug`   : 随机裁剪 + 水平翻转 (+ 可选颜色扰动)，用于训练集的数据增强。

`CatsDogsDataset` 把图片和标签组成样本，`DataLoader` 再把若干样本堆叠成一个 batch
（形状 `(N, 3, S, S)`），并负责打乱、并行读取。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import data_prep as dp  # noqa: E402

IMG_SIZE = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ 变换
def tf_stretch(size: int = IMG_SIZE):
    """直接拉伸成正方形（会改变宽高比）。"""
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def tf_resize_crop(size: int = IMG_SIZE):
    """等比缩放后中心裁剪（不改变宽高比）。"""
    return T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def tf_train_aug(size: int = IMG_SIZE, flip: bool = True, crop: bool = True,
                 color: bool = False, jitter: float = 0.25):
    """训练集变换，各增强项可开关，便于做对照实验。"""
    ops = []
    if crop:
        ops.append(T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.8, 1.25)))
    else:
        ops += [T.Resize(int(size * 1.14)), T.CenterCrop(size)]
    if flip:
        ops.append(T.RandomHorizontalFlip())
    if color:
        ops.append(T.ColorJitter(brightness=jitter, contrast=jitter,
                                 saturation=jitter, hue=0.05))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """把归一化后的 CHW tensor 还原到可显示的 0-1 HWC numpy。"""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = (t.cpu() * std + mean).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


# ------------------------------------------------------------------ Dataset
class CatsDogsDataset(Dataset):
    def __init__(self, split: str, transform=None, root: Path = dp.DATA_ROOT):
        self.rows = dp.load_split(split)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        img = Image.open(self.root / row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])

    @property
    def labels(self) -> list[int]:
        return [int(r["label"]) for r in self.rows]


def build_loader(split: str, transform, batch_size: int = 64, shuffle: bool | None = None,
                 num_workers: int = 4, root: Path = dp.DATA_ROOT) -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"
    ds = CatsDogsDataset(split, transform, root=root)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False, drop_last=False)


if __name__ == "__main__":
    loader = build_loader("train", tf_train_aug(), batch_size=8, num_workers=0)
    x, y = next(iter(loader))
    print("一个 batch 的 tensor 形状:", tuple(x.shape), "dtype:", x.dtype,
          " min/max:", round(float(x.min()), 3), round(float(x.max()), 3))
    print("对应标签:", y.tolist())
