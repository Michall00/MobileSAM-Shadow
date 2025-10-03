from typing import List, Tuple, Dict, Optional
import os, glob, random
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

def _to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")

def _to_bin_mask(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        img = img.convert("L")
    return img.point(lambda p: 255 if p >= 128 else 0, mode="L")

def _sam_resize_pad_hw(h: int, w: int, target: int) -> Tuple[float, int, int, int, int]:
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

class ObjPromptShadowDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        obj_masks_dir: str,
        target_masks_dir: str,
        size: int = 1024,
        pos_points_range: Tuple[int,int] = (1,3),
        neg_points_range: Tuple[int,int] = (1,3),
        scenario_probs: Tuple[float,float,float] = (0.5, 0.25, 0.25),
        box_from: str = "object",  # "object" or "target"
        box_jitter: float = 0.1,
        photometric_aug: bool = False,
        seed: Optional[int] = None,
        return_obj_mask: bool = False,
    ) -> None:
        self.images_dir = images_dir
        self.obj_masks_dir = obj_masks_dir
        self.target_masks_dir = target_masks_dir
        self.size = size
        self.pos_range = pos_points_range
        self.neg_range = neg_points_range
        self.scenario_probs = scenario_probs
        self.box_from = box_from
        self.box_jitter = box_jitter
        self.photometric_aug = photometric_aug
        self.return_obj_mask = return_obj_mask
        if seed is not None:
            random.seed(seed); np.random.seed(seed)

        img_paths = sorted(glob.glob(os.path.join(images_dir, "**", "*.*"), recursive=True))
        valid_ext = {".jpg",".jpeg",".png",".bmp",".webp"}
        self.samples: List[Tuple[str,str,str]] = []
        print(f"Scanning {images_dir} for images...")
        print(f"Looking for object masks in {obj_masks_dir} and target masks in {target_masks_dir}...")
        print(f"Found {len(img_paths)} image files.")
        for ip in img_paths:
            ext = os.path.splitext(ip)[1].lower()
            if ext not in valid_ext:
                continue
            base = os.path.splitext(os.path.basename(ip))[0]
            obj_cands = [os.path.join(obj_masks_dir, base + ".png"), os.path.join(obj_masks_dir, base + ".jpg")]
            tgt_cands = [os.path.join(target_masks_dir, base + ".png"), os.path.join(target_masks_dir, base + ".jpg")]
            op = next((p for p in obj_cands if os.path.isfile(p)), None)
            tp = next((p for p in tgt_cands if os.path.isfile(p)), None)
            if op and tp:
                self.samples.append((ip, op, tp))
        if not self.samples:
            raise RuntimeError("No (image, object_mask, target_mask) triplets found.")

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_photometric(self, img: Image.Image) -> Image.Image:
        if not self.photometric_aug:
            return img
        b = 1.0 + random.uniform(-0.15, 0.15)  # brightness
        c = 1.0 + random.uniform(-0.15, 0.15)  # contrast
        g = 1.0 + random.uniform(-0.10, 0.10)  # gamma
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.clip(arr ** g, 0, 1.0)
        mean = arr.mean(axis=(0,1), keepdims=True)
        arr = (arr - mean) * c + mean
        arr = np.clip(arr * b, 0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        ipath, obj_mpath, tgt_mpath = self.samples[idx]
        img = _to_rgb(Image.open(ipath))
        obj_mask = _to_bin_mask(Image.open(obj_mpath))
        tgt_mask = _to_bin_mask(Image.open(tgt_mpath))

        img = self._apply_photometric(img)

        w, h = img.size
        _, nh, nw, _, _ = _sam_resize_pad_hw(h, w, self.size)
        img_rp = _resize_pad_rgb(img, self.size)
        obj_rp = _resize_pad_mask(obj_mask, self.size)   # padding must be 0 inside
        tgt_rp = _resize_pad_mask(tgt_mask, self.size)

        obj_np = (np.asarray(obj_rp, dtype=np.uint8) > 0).astype(np.uint8)
        tgt_np = (np.asarray(tgt_rp, dtype=np.uint8) > 0).astype(np.uint8)

        pos = _sample_pos_from_object(obj_np, k=1)
        if pos.size == 0:
            rx = np.random.randint(0, max(1, nw))
            ry = np.random.randint(0, max(1, nh))
            pos = np.array([[rx, ry]], dtype=np.int32)

        points_xy = pos.astype(np.int32)          # [1,2] w przestrzeni size×size
        labels = np.ones((1,), dtype=np.int32)    # [1] same pozytywne
        box_xyxy = np.empty((0,), dtype=np.int32) # brak bboxa

        img_t = _normalize_sam(_img_to_tensor(img_rp)).float()
        tgt_t = _mask_to_tensor(tgt_rp).float()

        sample: Dict[str, Tensor] = {
            "image": img_t,                                   # [3,size,size]
            "mask": tgt_t,                                    # [1,size,size]
            "points": torch.from_numpy(points_xy),            # [1,2] int32
            "point_labels": torch.from_numpy(labels),         # [1]   int32
            "boxes": torch.from_numpy(box_xyxy),              # [0]
            "obj_mask": torch.from_numpy(obj_np[None].astype(np.float32)), # [1,size,size]
        }
        return sample



def two_mask_collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    points = [b["points"] for b in batch]
    point_labels = [b["point_labels"] for b in batch]
    boxes = [b["boxes"] for b in batch]
    out: Dict[str, Tensor] = {"image": images, "mask": masks, "points": points, "point_labels": point_labels, "boxes": boxes}
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


if __name__ == "__main__":
    from typing import Tuple
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import torch
    from torch.utils.data import DataLoader

    from mobile_sam.utils.dataset_utils import show_sample

    ds = ObjPromptShadowDataset(
        images_dir="/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/images",
        obj_masks_dir="/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/object_masks",
        target_masks_dir="/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/masks",
        size=1024,
        box_from="object",
        photometric_aug=True,
        seed=42,
        return_obj_mask=True,
    )

    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, collate_fn=two_mask_collate)
    batch = next(iter(loader))

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
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(x)
    plt.title("DEBUG: Obraz po denormalizacji (ręcznie)")
    plt.axis("off")
    plt.show()

    def _split_points(points: torch.Tensor, labels: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
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
            ax.scatter(pos[:, 0], pos[:, 1], s=40, marker="o")
        if neg.size > 0:
            ax.scatter(neg[:, 0], neg[:, 1], s=40, marker="x")
        if box.numel() == 4:
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
            rect = Rectangle((x0, y0), max(0, x1 - x0), max(0, y1 - y0), fill=False, linewidth=2)
            ax.add_patch(rect)
        ax.set_title("Overlay: OBJECT mask + points (+ box)")
        
        ax.set_xlim([0, W])
        ax.set_ylim([H, 0])
        ax.axis("off")
        plt.tight_layout(); plt.show()

    show_object_prompt_sample(batch, index=0)
