import numpy as np
from typing import Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def sample_positive_points(mask: np.ndarray, num_points: int = 1) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.empty((0,2), dtype=np.int32)
    idx = np.random.choice(len(xs), size=min(num_points, len(xs)), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1)  # (N,2) x,y


def sample_negative_points(mask: np.ndarray, num_points: int = 1) -> np.ndarray:
    ys, xs = np.where(mask == 0)
    if len(xs) == 0:
        return np.empty((0,2), dtype=np.int32)
    idx = np.random.choice(len(xs), size=num_points, replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1)


def sample_box(mask: np.ndarray, jitter: float = 0.1) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([0,0,0,0], dtype=np.int32)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    w, h = x_max - x_min, y_max - y_min
    jx, jy = int(jitter * w), int(jitter * h)
    return np.array([
        max(0, x_min - np.random.randint(0, jx+1)),
        max(0, y_min - np.random.randint(0, jy+1)),
        x_max + np.random.randint(0, jx+1),
        y_max + np.random.randint(0, jy+1)
    ], dtype=np.int32)


def generate_prompts(mask: np.ndarray) -> dict:
    r = np.random.rand()
    if r < 0.5:
        pts = sample_positive_points(mask, np.random.randint(1,4))
        labels = np.ones(len(pts), dtype=np.int32)
        return {"points": pts, "labels": labels, "boxes": None}
    elif r < 0.75:
        pos = sample_positive_points(mask, np.random.randint(1,3))
        neg = sample_negative_points(mask, np.random.randint(1,3))
        pts = np.vstack([pos, neg])
        labels = np.hstack([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int32)
        return {"points": pts, "labels": labels, "boxes": None}
    else:
        box = sample_box(mask)
        return {"points": None, "labels": None, "boxes": box}
    

def _sam_denormalize(img_t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(img_t.device)
    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(img_t.device)
    x = (img_t * std + mean) / 255.0
    x = x.clamp(0.0, 1.0)
    return x.permute(1, 2, 0).cpu().numpy()

def _split_points_by_label(points: torch.Tensor, labels: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    if points.numel() == 0 or labels.numel() == 0:
        return np.zeros((0,2), dtype=np.int32), np.zeros((0,2), dtype=np.int32)
    pts = points.cpu().numpy()
    lbs = labels.cpu().numpy().astype(np.int32)
    pos = pts[lbs == 1] if (lbs == 1).any() else np.zeros((0,2), dtype=np.int32)
    neg = pts[lbs == 0] if (lbs == 0).any() else np.zeros((0,2), dtype=np.int32)
    return pos, neg

def show_sample(batch: dict, index: int = 0) -> None:
    img_t = batch["image"][index]          # [3,H,W] normalized for SAM
    mask_t = batch["mask"][index, 0]       # [H,W] 0/1
    pts = batch["points"][index]           # [N,2] int32
    lbs = batch["point_labels"][index]     # [N] int32
    box = batch["boxes"][index]            # [4] or [0]

    img_np = _sam_denormalize(img_t)
    mask_np = mask_t.cpu().numpy()

    pos, neg = _split_points_by_label(pts, lbs)

    print(f"image: {batch['image'].shape}")
    print(f"mask: {batch['mask'].shape}")
    print(f"points[{index}]: {tuple(pts.shape)}")
    print(f"point_labels[{index}]: {tuple(lbs.shape)}")
    print(f"boxes[{index}]: {tuple(box.shape)}")

    H, W = mask_np.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Image"); axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Mask"); axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(mask_np, cmap="jet", alpha=0.35)
    if pos.size > 0:
        axes[2].scatter(pos[:, 0], pos[:, 1], s=40, marker="o")
    if neg.size > 0:
        axes[2].scatter(neg[:, 0], neg[:, 1], s=40, marker="x")
    if box.numel() == 4:
        x0, y0, x1, y1 = [int(v) for v in box.tolist()]
        w, h = max(0, x1 - x0), max(0, y1 - y0)
        rect = Rectangle((x0, y0), w, h, fill=False, linewidth=2)
        axes[2].add_patch(rect)
    axes[2].set_title("Overlay: mask + points (+ box)")
    axes[2].set_xlim([0, W]); axes[2].set_ylim([H, 0]); axes[2].axis("off")
    plt.tight_layout()
    plt.show()
