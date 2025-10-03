from typing import List, Tuple, Dict, Optional
import os
import glob
import random
import math
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

def _to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")

def _to_mask(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        img = img.convert("L")
    return img.point(lambda p: 255 if p >= 128 else 0, mode="L")

def _sam_resize_pad_hw(h: int, w: int, target: int) -> Tuple[float, int, int, int, int]:
    if h == 0 or w == 0:
        raise ValueError("Empty image.")
    scale = target / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    pad_h = target - new_h
    pad_w = target - new_w
    return scale, new_h, new_w, pad_h, pad_w

def _resize_pad_image(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    scale, new_h, new_w, pad_h, pad_w = _sam_resize_pad_hw(h, w, target)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    out = Image.new("RGB", (target, target))
    out.paste(img, (0, 0))
    return out

def _resize_pad_mask(mask: Image.Image, target: int) -> Image.Image:
    w, h = mask.size
    scale, new_h, new_w, pad_h, pad_w = _sam_resize_pad_hw(h, w, target)
    mask = mask.resize((new_w, new_h), Image.NEAREST)
    out = Image.new("L", (target, target))
    out.paste(mask, (0, 0))
    return out

def _transform_points(points_xy: np.ndarray, scale: float) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy
    pts = points_xy.astype(np.float32)
    pts *= scale
    return np.rint(pts).astype(np.int32)

def _transform_box(box_xyxy: np.ndarray, scale: float) -> np.ndarray:
    if box_xyxy.size == 0:
        return box_xyxy
    b = box_xyxy.astype(np.float32) * scale
    return np.rint(b).astype(np.int32)

def _gen_positive_points(mask_bin: np.ndarray, k: int) -> np.ndarray:
    ys, xs = np.where(mask_bin > 0)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    idx = np.random.choice(xs.size, size=min(k, xs.size), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.int32)

def _gen_negative_points(mask_bin: np.ndarray, k: int) -> np.ndarray:
    ys, xs = np.where(mask_bin == 0)
    if xs.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    idx = np.random.choice(xs.size, size=min(k, xs.size), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.int32)

def _gen_tight_box(mask_bin: np.ndarray, jitter_ratio: float = 0.1) -> np.ndarray:
    ys, xs = np.where(mask_bin > 0)
    if xs.size == 0:
        return np.empty((0,), dtype=np.int32)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    jx = max(1, int(round(jitter_ratio * w)))
    jy = max(1, int(round(jitter_ratio * h)))
    x0 = max(0, x0 - np.random.randint(0, jx + 1))
    y0 = max(0, y0 - np.random.randint(0, jy + 1))
    x1 = x1 + np.random.randint(0, jx + 1)
    y1 = y1 + np.random.randint(0, jy + 1)
    return np.array([x0, y0, x1, y1], dtype=np.int32)

def _normalize_sam(img: Tensor) -> Tensor:
    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
    return (img * 255.0 - mean) / std

def _img_to_tensor(img: Image.Image) -> Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

def _mask_to_tensor(mask: Image.Image) -> Tensor:
    arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
    arr = np.expand_dims(arr, 0)
    return torch.from_numpy(arr)

class SBUShadowDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        size: int = 1024,
        pos_points_range: Tuple[int, int] = (1, 3),
        neg_points_range: Tuple[int, int] = (1, 3),
        box_jitter: float = 0.1,
        scenario_probs: Tuple[float, float, float] = (0.5, 0.25, 0.25),
        photometric_aug: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.size = size
        self.pos_range = pos_points_range
        self.neg_range = neg_points_range
        self.box_jitter = box_jitter
        self.scenario_probs = scenario_probs
        self.photometric_aug = photometric_aug
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        img_glob = sorted(
            glob.glob(os.path.join(images_dir, "**", "*.*"), recursive=True)
        )
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.samples: List[Tuple[str, str]] = []
        for ipath in img_glob:
            ext = os.path.splitext(ipath)[1].lower()
            if ext not in valid_ext:
                continue
            fname = os.path.basename(ipath)
            mask_candidates = [
                os.path.join(masks_dir, fname),
                os.path.join(masks_dir, os.path.splitext(fname)[0] + ".png"),
                os.path.join(masks_dir, os.path.splitext(fname)[0] + ".jpg"),
            ]
            mpath = next((p for p in mask_candidates if os.path.isfile(p)), None)
            if mpath is not None:
                self.samples.append((ipath, mpath))
        if not self.samples:
            raise RuntimeError("No image/mask pairs found.")

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_photometric(self, img: Image.Image) -> Image.Image:
        if not self.photometric_aug:
            return img
        b = 1.0 + random.uniform(-0.15, 0.15)
        c = 1.0 + random.uniform(-0.15, 0.15)
        g = 1.0 + random.uniform(-0.10, 0.10)
        img = Image.fromarray(
            np.clip(np.asarray(img, dtype=np.float32) ** g, 0.0, 255.0).astype(np.uint8)
        )
        arr = np.asarray(img, dtype=np.float32)
        mean = arr.mean(axis=(0, 1), keepdims=True)
        arr = (arr - mean) * c + mean
        arr = arr * b
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _gen_prompts_raw(self, mask_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = random.random()
        p_only, p_mix, box_mode = self.scenario_probs
        if r < p_only:
            kpos = random.randint(self.pos_range[0], self.pos_range[1])
            pos = _gen_positive_points(mask_bin, kpos)
            labels = np.ones((len(pos),), dtype=np.int32)
            return pos, labels, np.empty((0,), dtype=np.int32)
        elif r < p_only + p_mix:
            kpos = random.randint(max(1, self.pos_range[0] - 0), self.pos_range[1])
            kneg = random.randint(self.neg_range[0], self.neg_range[1])
            pos = _gen_positive_points(mask_bin, kpos)
            neg = _gen_negative_points(mask_bin, kneg)
            pts = np.vstack([pos, neg]) if len(neg) > 0 else pos
            labels = np.hstack(
                [np.ones((len(pos),), dtype=np.int32), np.zeros((len(neg),), dtype=np.int32)]
            ) if len(neg) > 0 else np.ones((len(pos),), dtype=np.int32)
            return pts, labels, np.empty((0,), dtype=np.int32)
        else:
            box = _gen_tight_box(mask_bin, jitter_ratio=self.box_jitter)
            return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int32), box

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        ipath, mpath = self.samples[idx]

        img = _to_rgb(Image.open(ipath))
        mask = _to_mask(Image.open(mpath))

        img = self._apply_photometric(img)

        w, h = img.size
        mask_np = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)

        pts_xy, pts_labels, box_xyxy = self._gen_prompts_raw(mask_np)

        scale, new_h, new_w, pad_h, pad_w = _sam_resize_pad_hw(h, w, self.size)
        img_rp = _resize_pad_image(img, self.size)
        mask_rp = _resize_pad_mask(mask, self.size)

        pts_xy_t = _transform_points(pts_xy, scale)
        box_xyxy_t = _transform_box(box_xyxy, scale)

        img_t = _img_to_tensor(img_rp)             # [3,H,W] in [0,1]
        img_t = _normalize_sam(img_t).float()      # SAM norm
        mask_t = _mask_to_tensor(mask_rp).float()  # [1,H,W] {0,1}

        points = torch.from_numpy(pts_xy_t.astype(np.int32)) if pts_xy_t.size else torch.zeros((0, 2), dtype=torch.int32)
        point_labels = torch.from_numpy(pts_labels.astype(np.int32)) if pts_labels.size else torch.zeros((0,), dtype=torch.int32)
        boxes = torch.from_numpy(box_xyxy_t.astype(np.int32)) if box_xyxy_t.size else torch.zeros((0,), dtype=torch.int32)

        sample: Dict[str, Tensor] = {
            "image": img_t,                  # [3,1024,1024] float32
            "mask": mask_t,                  # [1,1024,1024] float32
            "points": points,                # [N,2] int32 or [0,2]
            "point_labels": point_labels,    # [N] int32 or [0]
            "boxes": boxes,                  # [4] int32 or [0]
        }
        return sample

def sbu_collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    # variable-size prompts: pack as lists of tensors
    points = [b["points"] for b in batch]
    point_labels = [b["point_labels"] for b in batch]
    boxes = [b["boxes"] for b in batch]
    return {
        "image": images,
        "mask": masks,
        "points": points,
        "point_labels": point_labels,
        "boxes": boxes,
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from mobile_sam.utils.dataset_utils import show_sample

    ds = SBUShadowDataset(
        images_dir="/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/SBU-shadow/SBU-Test/ShadowImages",
        masks_dir="/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/SBU-shadow/SBU-Test/ShadowMasks",
        size=1024,
        photometric_aug=True,
        seed=42,
    )

    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, collate_fn=sbu_collate)
    batch = next(iter(loader))

    print("image:", batch["image"].shape)         # [B,3,1024,1024]
    print("mask:", batch["mask"].shape)           # [B,1,1024,1024]
    print("points lens:", [p.shape for p in batch["points"]])
    print("point_labels lens:", [l.shape for l in batch["point_labels"]])
    print("boxes lens:", [b.shape for b in batch["boxes"]])

    show_sample(batch, index=0)