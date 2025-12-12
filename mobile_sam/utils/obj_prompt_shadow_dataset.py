from typing import Optional
import os, glob, random
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path

from mobile_sam.utils.dataset_utils import (
    _to_rgb,
    _to_bin_mask,
    _sam_resize_pad_hw,
    _resize_pad_rgb,
    _resize_pad_mask,
    _normalize_sam,
    _img_to_tensor,
    _mask_to_tensor,
    sample_positive_points,
    sample_negative_points,
    sample_box,
    _to_list,
    _sample_pos_from_object,
    _tight_bbox,
    AugmentationConfig,
    TASK_SHADOW
)


class ObjPromptShadowDataset(Dataset):
    def __init__(
        self,
        images_dir: str | list[str],
        object_masks_dir: str | list[str],
        target_masks_dir: str | list[str],
        size: int = 1024,
        pos_points_range: tuple[int,int] = (1,3),
        neg_points_range: tuple[int,int] = (1,3),
        box_from: str = "object",  # "object" or "target"
        box_jitter: float = 0.1,
        seed: Optional[int] = None,
        return_obj_mask: bool = False,
        augmenter: Optional[A.Compose] = None
    ) -> None:
        self.images_dir_list = [Path(p) for p in _to_list(images_dir)]
        self.object_masks_dir_list = [Path(p) for p in _to_list(object_masks_dir)]
        self.target_masks_dir_list = [Path(p) for p in _to_list(target_masks_dir)]
        self.size = size
        self.pos_range = pos_points_range
        self.neg_range = neg_points_range
        self.box_from = box_from
        self.box_jitter = box_jitter
        self.augmenter = augmenter
        self.return_obj_mask = return_obj_mask
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        img_paths = sorted(p for img_dir in self.images_dir_list for p in img_dir.rglob("*.*"))
        valid_ext = {".jpg",".jpeg",".png",".bmp",".webp"}
        self.samples: list[tuple[str,str,str,str]] = []  # (img, obj_mask, tgt_mask, mode)
        print(f"Scanning {images_dir} for images...")
        print(f"Looking for object masks in {object_masks_dir} and target masks in {target_masks_dir}...")
        print(f"Found {len(img_paths)} image files.")

        for img_dir, obj_dir, tgt_dir in zip(self.images_dir_list, self.object_masks_dir_list, self.target_masks_dir_list):
            img_paths = sorted(p for p in img_dir.rglob("*.*"))
            for ip in img_paths:
                ext = ip.suffix.lower()
                if ext not in valid_ext:
                    continue
                base = ip.stem
                obj_cands = [obj_dir / f"{base}.png", obj_dir / f"{base}.jpg"]
                tgt_cands = [tgt_dir / f"{base}.png", tgt_dir / f"{base}.jpg"]
                op = next((p for p in obj_cands if p.is_file()), None)
                tp = next((p for p in tgt_cands if p.is_file()), None)
                if op and tp:
                    self.samples.append((ip, op, tp, "point"))
                    self.samples.append((str(ip), str(op), str(tp), "box"))

        if not self.samples:
            raise RuntimeError("No (image, object_mask, target_mask) triplets found.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ipath, obj_mpath, tgt_mpath, mode = self.samples[idx]
        img = _to_rgb(Image.open(ipath))
        obj_mask = _to_bin_mask(Image.open(obj_mpath))
        tgt_mask = _to_bin_mask(Image.open(tgt_mpath))

        if self.augmenter is not None:
            img, obj_mask, tgt_mask = self.augmenter(np.array(img), np.array(obj_mask), np.array(tgt_mask))
            img = self.augmenter.to_pil(img)
            obj_mask = self.augmenter.to_pil(obj_mask)
            tgt_mask = self.augmenter.to_pil(tgt_mask)

        w, h = img.size
        _, nh, nw, _, _ = _sam_resize_pad_hw(h, w, self.size)
        img_rp = _resize_pad_rgb(img, self.size)
        obj_rp = _resize_pad_mask(obj_mask, self.size)
        tgt_rp = _resize_pad_mask(tgt_mask, self.size)

        obj_np = (np.asarray(obj_rp, dtype=np.uint8) > 0).astype(np.uint8)
        tgt_np = (np.asarray(tgt_rp, dtype=np.uint8) > 0).astype(np.uint8)

        img_t = _normalize_sam(_img_to_tensor(img_rp)).float()
        tgt_t = _mask_to_tensor(tgt_rp).float()

        if mode == "point":
            pos = _sample_pos_from_object(obj_np, k=1)
            if pos.size == 0:
                rx = np.random.randint(0, max(1, nw))
                ry = np.random.randint(0, max(1, nh))
                pos = np.array([[rx, ry]], dtype=np.int32)

            points_xy = pos.astype(np.int32)
            labels = np.ones((1,), dtype=np.int32)
            boxes = np.empty((0,), dtype=np.int32)
        else:
            points_xy = np.empty((0, 2), dtype=np.int32)
            labels = np.empty((0,), dtype=np.int32)
            bbox = _tight_bbox(obj_np, jitter=self.box_jitter)
            boxes = bbox.astype(np.int32) if bbox.size == 4 else np.empty((0,), dtype=np.int32)

        return {
            "image": img_t,
            "mask": tgt_t,
            "points": torch.from_numpy(points_xy),
            "point_labels": torch.from_numpy(labels),
            "boxes": torch.from_numpy(boxes),
            "obj_mask": torch.from_numpy(obj_np[None].astype(np.float32)),
            "task_id": TASK_SHADOW
        }


def two_mask_collate(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    points = [b["points"] for b in batch]
    point_labels = [b["point_labels"] for b in batch]
    boxes = [b["boxes"] for b in batch]
    out: dict[str, Tensor] = {"image": images, "mask": masks, "points": points, "point_labels": point_labels, "boxes": boxes}
    if "obj_mask" in batch[0]:
        out["obj_mask"] = torch.stack([b["obj_mask"] for b in batch], dim=0)
    return out


def sam_denormalize(img_t: torch.Tensor) -> np.ndarray:
    x = img_t.detach().cpu().float().numpy()           # [3,H,W]
    mean = np.array([123.675, 116.28, 103.53], np.float32)[:, None, None]
    std  = np.array([58.395, 57.12, 57.375], np.float32)[:, None, None]
    x = x * std + mean
    x = np.clip(x, 0.0, 255.0)
    x = x.transpose(1, 2, 0).astype(np.uint8)
    return x



def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import torch
    from torch.utils.data import DataLoader

    from mobile_sam.utils.dataset_utils import show_sample

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

    ds = ObjPromptShadowDataset(
        images_dir=[
            "/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/images",
            "/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/images_aug/flip",
        ],
        obj_masks_dir=[
            "/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/object_masks",
            "/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/object_masks_aug/flip",
        ],
        target_masks_dir=[
            "/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/masks",
            "/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/masks_aug/flip",
        ],
        size=1024,
        box_from="object",
        seed=42,
        return_obj_mask=True,
        augmenter=None
    )

    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, collate_fn=two_mask_collate)
    batch = next(iter(loader))

    print(f"Batches: {len(loader)}")
    print("image:", batch["image"].shape)                # [B,3,1024,1024]
    print("mask (target):", batch["mask"].shape)         # [B,1,1024,1024]
    if "obj_mask" in batch:
        print("obj_mask:", batch["obj_mask"].shape)      # [B,1,1024,1024]
    print("points lens:", [p.shape for p in batch["points"]])
    print("point_labels lens:", [l.shape for l in batch["point_labels"]])
    print("boxes lens:", [b.shape for b in batch["boxes"]])
    print("obj_mask lens:", [o.shape for o in batch["obj_mask"]])


    ipath = ds.samples[0][0]
    img = _to_rgb(Image.open(ipath))
    import matplotlib.pyplot as plt

    for batch in loader:
        show_sample(batch, index=0)

    img_t = batch["image"][0]
    mean = np.array([123.675, 116.28, 103.53], np.float32)[:, None, None]
    std  = np.array([58.395, 57.12, 57.375], np.float32)[:, None, None]
    x = img_t.detach().cpu().float().numpy()
    x = x * std + mean
    x = np.clip(x, 0, 255).astype(np.uint8)
    x = x.transpose(1,2,0)

    plt.figure()
    plt.imshow(x)
    plt.title("DEBUG: Image after denormalization (manual)")
    plt.axis("off")
    plt.show()

    def _split_points(points: torch.Tensor, labels: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if points.numel() == 0 or labels.numel() == 0:
            return np.zeros((0,2), dtype=np.int32), np.zeros((0,2), dtype=np.int32)
        pts = points.cpu().numpy()
        lbs = labels.cpu().numpy().astype(np.int32)
        pos = pts[lbs == 1] if (lbs == 1).any() else np.zeros((0,2), dtype=np.int32)
        neg = pts[lbs == 0] if (lbs == 0).any() else np.zeros((0,2), dtype=np.int32)
        return pos, neg

    def show_object_prompt_sample(batch: dict, index: int = 0) -> None:
        img_t = batch["image"][index]
        obj = batch["obj_mask"][index, 0] if "obj_mask" in batch else batch["mask"][index, 0]
        pts = batch["points"][index]
        lbs = batch["point_labels"][index]
        box = batch["boxes"][index]

        img_np = sam_denormalize(img_t)
        obj_np = obj.cpu().numpy() > 0.5
        pos, neg = _split_points(pts, lbs)

        H, W = obj_np.shape
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img_np)
        ax.imshow(obj_np, cmap="jet", alpha=0.35)
        if pos.size > 0:
            ax.scatter(pos[:, 0], pos[:, 1], s=40, marker="o", label="Positive Points")
        if neg.size > 0:
            ax.scatter(neg[:, 0], neg[:, 1], s=40, marker="x", label="Negative Points")
        if box.numel() == 4:
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
            rect = Rectangle((x0, y0), max(0, x1 - x0), max(0, y1 - y0), fill=False, edgecolor="red", linewidth=2, label="Bounding Box")
            ax.add_patch(rect)
        ax.set_title("Overlay: OBJECT mask + points (+ box)")
        
        ax.set_xlim([0, W])
        ax.set_ylim([H, 0])
        ax.axis("off")
        ax.legend(loc="upper right")
        plt.tight_layout(); plt.show()

    show_object_prompt_sample(batch, index=0)


if __name__ == "__main__":
    main()