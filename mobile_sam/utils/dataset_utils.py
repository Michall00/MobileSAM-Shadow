import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from torch import Tensor
import albumentations as A

TASK_SHADOW = 0
TASK_REFLECTION = 1
TASK_NORMAL = 2

class AugmentationConfig:
    def __init__(self, cfg_aug: dict):
        self.transforms = []
        p = cfg_aug.get("prob", 0.5)

        if cfg_aug.get("flip", False):
            self.transforms.append(A.HorizontalFlip(p=p))

        if cfg_aug.get("rotate", False):
            self.transforms.append(A.Rotate(limit=cfg_aug.get("rotate_limit", 15), border_mode=0, p=p))

        if cfg_aug.get("brightness", False):
            self.transforms.append(A.RandomBrightnessContrast(p=p))

        if cfg_aug.get("blur", False):
            self.transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=p))

        if cfg_aug.get("noise", False):
             self.transforms.append(A.GaussNoise(p=p))

        self.pipeline = A.Compose(
            self.transforms,
            additional_targets={"obj_mask": "mask", "tgt_mask": "mask"}
        )
    def __call__(self, image: Image.Image, obj_mask: np.ndarray, tgt_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        augmented = self.pipeline(image=image, obj_mask=obj_mask, tgt_mask=tgt_mask)
        return augmented["image"], augmented["obj_mask"], augmented["tgt_mask"]
    
    @staticmethod
    def to_pil(image: np.ndarray | Image.Image) -> Image.Image:
        return image if isinstance(image, Image.Image) else Image.fromarray(image)


def two_mask_collate(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    points = [b["points"] for b in batch]
    point_labels = [b["point_labels"] for b in batch]
    boxes = [b["boxes"] for b in batch]
    out: dict[str, Tensor] = {"image": images, "mask": masks, "points": points, "point_labels": point_labels, "boxes": boxes}
    
    if "obj_mask" in batch[0]:
        out["obj_mask"] = torch.stack([b["obj_mask"] for b in batch], dim=0)

    if "task_id" in batch[0]:
        out["task_id"] = torch.tensor([b["task_id"] for b in batch], dtype=torch.long)
        
    return out


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
    

def sam_denormalize_float(img_t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(img_t.device)
    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(img_t.device)
    x = (img_t * std + mean) / 255.0
    x = x.clamp(0.0, 1.0)
    return x.permute(1, 2, 0).cpu().numpy()

def _split_points_by_label(points: torch.Tensor, labels: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
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

    img_np = sam_denormalize_float(img_t)
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


def _to_list(x: str | list[str]) -> list[str]:
    return [x] if isinstance(x, str) else x

def _to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")

def _to_bin_mask(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        img = img.convert("L")
    return img.point(lambda p: 255 if p >= 128 else 0, mode="L")

def _sam_resize_pad_hw(h: int, w: int, target: int) -> tuple[float, int, int, int, int]:
    scale = target / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    return scale, new_h, new_w, target - new_h, target - new_w

def _resize_pad_rgb(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    scale, nh, nw, _, _ = _sam_resize_pad_hw(h, w, target)
    img = img.resize((nw, nh), Image.BILINEAR)
    out = Image.new("RGB", (target, target), (255, 255, 255))
    out.paste(img, (0, 0))
    return out

def _resize_pad_mask(mask: Image.Image, target: int) -> Image.Image:
    w, h = mask.size
    scale, nh, nw, _, _ = _sam_resize_pad_hw(h, w, target)
    mask = mask.resize((nw, nh), Image.NEAREST)
    out = Image.new("L", (target, target), 0)
    out.paste(mask, (0, 0))
    return out

def _normalize_sam(img_t: Tensor) -> Tensor:
    mean = torch.tensor([123.675, 116.28, 103.53]).view(3,1,1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(3,1,1)
    return (img_t * 255.0 - mean) / std

def _img_to_tensor(img: Image.Image) -> Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

def _mask_to_tensor(mask: Image.Image) -> Tensor:
    arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
    arr = np.expand_dims(arr, 0)
    return torch.from_numpy(arr)

def _sample_pos_from_object(obj_mask_np: np.ndarray, k: int) -> np.ndarray:
    ys, xs = np.where(obj_mask_np > 0)
    if xs.size == 0:
        return np.empty((0,2), dtype=np.int32)
    idx = np.random.choice(xs.size, size=min(k, xs.size), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.int32)

def _tight_bbox(mask_np: np.ndarray, jitter: float = 0.1) -> np.ndarray:
    ys, xs = np.where(mask_np > 0)
    if xs.size == 0:
        return np.empty((0,), dtype=np.int32)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    jx = max(1, int(round(jitter * w)))
    jy = max(1, int(round(jitter * h)))
    x0 = max(0, x0 - np.random.randint(0, jx + 1))
    y0 = max(0, y0 - np.random.randint(0, jy + 1))
    x1 = x1 + np.random.randint(0, jx + 1)
    y1 = y1 + np.random.randint(0, jy + 1)
    return np.array([x0, y0, x1, y1], dtype=np.int32)
