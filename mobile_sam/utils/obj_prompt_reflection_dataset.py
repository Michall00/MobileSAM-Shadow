from typing import Optional
import random
import torch
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from matplotlib.patches import Rectangle

from mobile_sam.utils.dataset_utils import (
    _to_rgb,
    _to_bin_mask,
    _sam_resize_pad_hw,
    _resize_pad_rgb,
    _resize_pad_mask,
    _normalize_sam,
    _img_to_tensor,
    _mask_to_tensor,
    _to_list,
    _sample_pos_from_object,
    two_mask_collate,
    _tight_bbox,
    AugmentationConfig,
    TASK_REFLECTION
)

class ObjPromptReflectionDataset(Dataset):
    def __init__(
        self,
        images_dir: str | list[str],
        object_masks_dir: str | list[str],
        target_masks_dir: str | list[str],
        size: int = 1024,
        box_jitter: float = 0.1,
        seed: Optional[int] = None,
        return_obj_mask: bool = False,
        augmenter: Optional[A.Compose] = None,
        task_id: int = TASK_REFLECTION
    ) -> None:
        self.images_dir_list = [Path(p) for p in _to_list(images_dir)]
        self.object_masks_dir_list = [Path(p) for p in _to_list(object_masks_dir)]
        self.target_masks_dir_list = [Path(p) for p in _to_list(target_masks_dir)]
        self.size = size
        self.box_jitter = box_jitter
        self.augmenter = augmenter
        self.return_obj_mask = return_obj_mask
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        valid_ext = {".jpg",".jpeg",".png",".bmp",".webp"}
        self.samples = []
        
        print(f"Scanning for Reflection dataset...")
        
        for img_dir, obj_dir, tgt_dir in zip(self.images_dir_list, self.object_masks_dir_list, self.target_masks_dir_list):
            img_paths = sorted(p for p in img_dir.rglob("*.*"))
            for ip in img_paths:
                if ip.suffix.lower() not in valid_ext: continue
                
                base = ip.stem
                op = next((p for p in [obj_dir / f"{base}.png", obj_dir / f"{base}.jpg"] if p.is_file()), None)
                tp = next((p for p in [tgt_dir / f"{base}.png", tgt_dir / f"{base}.jpg"] if p.is_file()), None)
                
                if op and tp:
                    self.samples.append((str(ip), str(op), str(tp)))

        if not self.samples:
            raise RuntimeError("No triplets found for Reflection dataset.")
        print(f"Found {len(self.samples)} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ipath, obj_mpath, refl_mpath = self.samples[idx]
        
        img = _to_rgb(Image.open(ipath))
        obj_mask = _to_bin_mask(Image.open(obj_mpath))
        refl_mask = _to_bin_mask(Image.open(refl_mpath))

        if self.augmenter is not None:
            img, obj_mask, refl_mask = self.augmenter(np.array(img), np.array(obj_mask), np.array(refl_mask))
            img = self.augmenter.to_pil(img)
            obj_mask = self.augmenter.to_pil(obj_mask)
            refl_mask = self.augmenter.to_pil(refl_mask)

        img_rp = _resize_pad_rgb(img, self.size)
        obj_rp = _resize_pad_mask(obj_mask, self.size)
        refl_rp = _resize_pad_mask(refl_mask, self.size)

        obj_np = (np.asarray(obj_rp, dtype=np.uint8) > 0).astype(np.uint8)
        
        bbox = _tight_bbox(obj_np, jitter=self.box_jitter)
        boxes = bbox.astype(np.int32) if bbox.size == 4 else np.empty((0,), dtype=np.int32)
        
        img_t = _normalize_sam(_img_to_tensor(img_rp)).float()
        tgt_t = _mask_to_tensor(refl_rp).float()

        # MobileSAM prompt format
        points_xy = np.empty((0, 2), dtype=np.int32)
        labels = np.empty((0,), dtype=np.int32)

        return {
            "image": img_t,
            "mask": tgt_t,
            "boxes": torch.from_numpy(boxes),
            "points": torch.from_numpy(points_xy),
            "point_labels": torch.from_numpy(labels),
            "obj_mask": torch.from_numpy(obj_np[None].astype(np.float32)),
            "task_id": TASK_REFLECTION
        }


def sam_denormalize_int(img_t: torch.Tensor) -> np.ndarray:
    x = img_t.detach().cpu().float().numpy()           
    mean = np.array([123.675, 116.28, 103.53], np.float32)[:, None, None]
    std  = np.array([58.395, 57.12, 57.375], np.float32)[:, None, None]
    x = x * std + mean
    x = np.clip(x, 0.0, 255.0)
    x = x.transpose(1, 2, 0).astype(np.uint8)
    return x

def _split_points(points: torch.Tensor, labels: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    if points.numel() == 0 or labels.numel() == 0:
        return np.zeros((0,2), dtype=np.int32), np.zeros((0,2), dtype=np.int32)
    pts = points.cpu().numpy()
    lbs = labels.cpu().numpy().astype(np.int32)
    pos = pts[lbs == 1] if (lbs == 1).any() else np.zeros((0,2), dtype=np.int32)
    neg = pts[lbs == 0] if (lbs == 0).any() else np.zeros((0,2), dtype=np.int32)
    return pos, neg

def show_object_prompt_sample(batch: dict, index: int = 0, title_suffix: str = "") -> None:
    img_t = batch["image"][index]
    target_mask = batch["mask"][index, 0]
    obj_mask = batch["obj_mask"][index, 0] if "obj_mask" in batch else None
    
    pts = batch["points"][index]
    lbs = batch["point_labels"][index]
    box = batch["boxes"][index]

    img_np = sam_denormalize_int(img_t)
    target_np = target_mask.cpu().numpy() > 0.5
    
    pos, neg = _split_points(pts, lbs)

    H, W = target_np.shape
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    ax = axes[0]
    ax.imshow(img_np)
    ax.imshow(target_np, cmap="jet", alpha=0.4) 
    ax.set_title(f"Target (Reflection + Object)\n{title_suffix}")
    ax.axis("off")

    ax = axes[1]
    ax.imshow(img_np)
    
    if obj_mask is not None:
        obj_np = obj_mask.cpu().numpy() > 0.5
        green_mask = np.zeros((H, W, 4))
        green_mask[obj_np, 1] = 1.0
        green_mask[obj_np, 3] = 0.3
        ax.imshow(green_mask)

    if pos.size > 0:
        ax.scatter(pos[:, 0], pos[:, 1], c='lime', s=100, marker="*", label="Positive (Click)", edgecolors='black')
    if neg.size > 0:
        ax.scatter(neg[:, 0], neg[:, 1], c='red', s=100, marker="x", label="Negative", edgecolors='black')
    
    if box.numel() == 4:
        x0, y0, x1, y1 = [int(v) for v in box.tolist()]
        rect = Rectangle((x0, y0), max(0, x1 - x0), max(0, y1 - y0), fill=False, edgecolor="yellow", linewidth=3, label="Bounding Box")
        ax.add_patch(rect)
        
    ax.set_title("Prompt Source (Object Only)")
    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])
    ax.axis("off")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

def main():
    aug_params = {
        "prob": 0.5,
        "flip": True,
        "brightness": True,
        "rotate": True,
        "rotate_limit": 15,
        "blur": True,
        "noise": False
    }
    augmenter = AugmentationConfig(aug_params)
    root_dir = "C:\\Users\\msado\\Studia\\MobileSAM\\MobileSAM-Shadow\\data\\DEROBA"
    
    ds = ObjPromptReflectionDataset(
        images_dir=[f"{root_dir}/images_without_reflection"],
        object_masks_dir=[f"{root_dir}/object_masks"],
        target_masks_dir=[f"{root_dir}/object_masks"],
        size=1024,
        box_jitter=0.1,
        seed=42,
        return_obj_mask=False,
        augmenter=augmenter
    )

    print(f"Dataset created. Total samples: {len(ds)}")

    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=two_mask_collate)
    
    try:
        batch = next(iter(loader))
    except Exception as e:
        print(f"Błąd podczas ładowania batcha: {e}")
        return

    print(f"Batch loaded!")
    print("Image shape:", batch["image"].shape)  # [B, 3, 1024, 1024]
    print("Mask shape:", batch["mask"].shape)    # [B, 1, 1024, 1024]
    
    for i in range(len(batch["image"])):
        print(f"Visualizing sample {i+1}/{len(batch['image'])}...")
        show_object_prompt_sample(batch, index=i, title_suffix=f"Sample {i}")

if __name__ == "__main__":
    main()