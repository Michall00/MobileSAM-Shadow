# train_mobilesam_shadow.py
from typing import Dict, List, Optional, Tuple
import os
import math
import time
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from PIL import Image

# Change these imports to your actual module/file names:
from mobile_sam.utils.dataset import SBUShadowDataset, sbu_collate  # <- your dataset from previous step
from mobile_sam.build_sam import sam_model_registry

def sam_denormalize(img_t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([123.675, 116.28, 103.53], device=img_t.device).view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device=img_t.device).view(3, 1, 1)
    x = (img_t * std + mean) / 255.0
    x = x.clamp(0.0, 1.0)
    x = (x.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return x

def color_overlay(img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = img.copy()
    overlay = np.zeros_like(out, dtype=np.uint8)
    overlay[mask.astype(bool)] = color
    out = (out.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha).astype(np.uint8)
    return out

def make_panel(img: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    left = Image.fromarray(img)
    mid = Image.fromarray(color_overlay(img, gt, (0, 255, 0)))
    right = Image.fromarray(color_overlay(img, pred, (255, 0, 0)))
    w, h = left.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(left, (0, 0))
    canvas.paste(mid, (w, 0))
    canvas.paste(right, (2 * w, 0))
    return canvas


class ShadowLoss(nn.Module):
    def __init__(self, pos_weight: float = 3.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    @staticmethod
    def dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num = 2 * (probs * target).sum(dim=(1, 2, 3)) + 1e-6
        den = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
        return 1.0 - (num / den).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, target) + self.dice(logits, target)


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).float()
    tp = (pred * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    iou = tp / max(1.0, (tp + fp + fn))
    prec = tp / max(1.0, (tp + fp))
    rec = tp / max(1.0, (tp + fn))
    f1 = 2 * prec * rec / max(1e-6, (prec + rec))
    ber = 0.5 * (fn / max(1.0, tp + fn) + fp / max(1.0, tn + fp))
    return {"iou": iou, "f1": f1, "ber": ber}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_non_encoder(model: nn.Module) -> List[nn.Parameter]:
    for p in getattr(model, "prompt_encoder").parameters():
        p.requires_grad = False
    for p in getattr(model, "mask_decoder").parameters():
        p.requires_grad = False
    enc_params = [p for p in getattr(model, "image_encoder").parameters()]
    for p in enc_params:
        p.requires_grad = True
    return enc_params


import torch
import torch.nn.functional as F
from typing import List


@torch.enable_grad()
def forward_mobile_sam(
    model: torch.nn.Module,
    images: torch.Tensor,                   # [B,3,H,W], already SAM-normalized and 1024x1024
    points: List[torch.Tensor],             # list of [Ni,2] int32 per-sample in 1024 grid
    point_labels: List[torch.Tensor],       # list of [Ni]   int32 per-sample (1=pos,0=neg)
    boxes: List[torch.Tensor],              # list of [4] xyxy int32 or empty tensor
    multimask_output: bool = False
) -> torch.Tensor:
    """
    Direct call to MobileSAM submodules:
      image_encoder -> prompt_encoder -> mask_decoder -> upsample to HxW.
    Returns logits of shape [B,1,H,W].
    """
    device = images.device
    B, _, H, W = images.shape
    assert H == 1024 and W == 1024, "Expected 1024x1024 inputs (match your dataset)."

    # 1) Encode image
    image_embeddings = model.image_encoder(images)  # [B,256,64,64] typically

    # 2) Positional enc. from prompt encoder (buffer moved with model.to(device))
    image_pe = model.prompt_encoder.get_dense_pe()  # [1,256,64,64]

    logits_out: List[torch.Tensor] = []

    for i in tqdm(range(B)):
        # Prepare prompts for sample i
        pts_i = points[i]
        lbs_i = point_labels[i]
        box_i = boxes[i]

        # None if empty
        pts_tuple = None
        if pts_i.numel() > 0:
            # PromptEncoder expects float coords and int labels with batch dim
            coords = pts_i.to(device=device, dtype=torch.float32).unsqueeze(0)       # [1,Ni,2]
            labels = lbs_i.to(device=device, dtype=torch.int64).unsqueeze(0)         # [1,Ni]
            pts_tuple = (coords, labels)

        box_tensor = None
        if box_i.numel() == 4:
            box_tensor = box_i.to(device=device, dtype=torch.float32).unsqueeze(0)   # [1,4]

        # 3) Encode prompts
        sparse_embeds, dense_embeds = model.prompt_encoder(
            points=pts_tuple,
            boxes=box_tensor,
            masks=None
        )  # sparse: [1,Np,256], dense: [1,256,64,64]

        # 4) Decode mask(s)
        lowres_logits, iou_pred = model.mask_decoder(
            image_embeddings=image_embeddings[i:i+1],     # [1,256,64,64]
            image_pe=image_pe,                             # [1,256,64,64]
            sparse_prompt_embeddings=sparse_embeds,        # [1,Np,256]
            dense_prompt_embeddings=dense_embeds,          # [1,256,64,64]
            multimask_output=multimask_output
        )  # -> [1,K,256,256], [1,K] where K=1 if multimask_output=False else 3

        # 5) Choose one mask (if multimask_output=True pick best by IoU)
        if lowres_logits.shape[1] > 1:
            best_idx = torch.argmax(iou_pred, dim=1)      # [1]
            chosen = lowres_logits[torch.arange(1, device=device), best_idx]  # [1,256,256]
        else:
            chosen = lowres_logits[:, 0]                   # [1,256,256]

        # 6) Upsample to input size
        chosen = F.interpolate(chosen.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)  # [1,1,1024,1024]
        logits_out.append(chosen)

    return torch.cat(logits_out, dim=0)  # [B,1,1024,1024]



class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        lr: float = 1e-4,
        weight_decay: float = 5e-2,
        max_epochs: int = 10,
        grad_clip: float = 1.0,
        amp: bool = True,
        device: str = "cuda",
        vis_dir: Optional[str] = None,
        vis_every: int = 5,
        vis_num: int = 8,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = ShadowLoss(pos_weight=3.0).to(device)
        enc_params = freeze_non_encoder(self.model)
        self.optimizer = torch.optim.AdamW(enc_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.amp = amp
        self.device = device
        enable_amp = bool(amp and self.device == "cuda")
        try:
            # PyTorch 2.x (nowe API) – pierwszy argument to device ('cuda'/'cpu')
            self.scaler = torch.amp.GradScaler(self.device, enabled=enable_amp)
        except TypeError:
            # Starsze warianty – bez parametru device
            self.scaler = torch.amp.GradScaler(enabled=enable_amp)
        self.vis_dir = vis_dir
        self.vis_every = vis_every
        self.vis_num = vis_num

    
    def save_predictions(self, epoch: int) -> None:
        if not self.vis_dir:
            return
        os.makedirs(self.vis_dir, exist_ok=True)
        loader = self.val_loader if self.val_loader is not None else self.train_loader
        self.model.eval()
        saved = 0
        with torch.no_grad():
            for batch in loader:
                print(f"Visualizing batch, saved {saved}/{self.vis_num} images...")
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                points = [p.to(self.device) for p in batch["points"]]
                labels = [l.to(self.device) for l in batch["point_labels"]]
                boxes = [b.to(self.device) for b in batch["boxes"]]
                logits = forward_mobile_sam(self.model, images, points, labels, boxes)
                probs = torch.sigmoid(logits)
                B = images.size(0)
                for i in range(B):
                    # if saved >= self.vis_num:
                    #     break
                    img_np = sam_denormalize(images[i])
                    gt = (masks[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                    pr = (probs[i, 0].detach().cpu().numpy() >= 0.5).astype(np.uint8)
                    panel = make_panel(img_np, gt, pr)
                    out_path = os.path.join(self.vis_dir, f"epoch_{epoch:03d}_{saved:03d}.png")
                    panel.save(out_path)
                    saved += 1
                # if saved >= self.vis_num:
                    # break
        self.model.train()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            points = [p.to(self.device) for p in batch["points"]]
            labels = [l.to(self.device) for l in batch["point_labels"]]
            boxes = [b.to(self.device) for b in batch["boxes"]]

            with torch.amp.autocast(self.device, enabled=(self.amp and self.device == "cuda")):
                logits = forward_mobile_sam(self.model, images, points, labels, boxes)
                loss = self.loss_fn(logits, masks)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.image_encoder.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            n_batches += 1

        return {"loss": total_loss / max(1, n_batches)}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None:
            return {"val_loss": math.nan, "val_iou": math.nan, "val_f1": math.nan, "val_ber": math.nan}
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        agg = {"iou": 0.0, "f1": 0.0, "ber": 0.0}
        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            points = [p.to(self.device) for p in batch["points"]]
            labels = [l.to(self.device) for l in batch["point_labels"]]
            boxes = [b.to(self.device) for b in batch["boxes"]]

            logits = forward_mobile_sam(self.model, images, points, labels, boxes)
            loss = self.loss_fn(logits, masks)
            total_loss += float(loss.item()); n_batches += 1
            m = compute_metrics(logits, masks, thr=0.5)
            for k in agg: agg[k] += m[k]
        for k in agg: agg[k] /= max(1, n_batches)
        return {"val_loss": total_loss / max(1, n_batches), "val_iou": agg["iou"], "val_f1": agg["f1"], "val_ber": agg["ber"]}

    def fit(self, ckpt_path: str) -> None:
        best_f1 = -1.0
        for epoch in tqdm(range(1, self.max_epochs + 1)):
            t0 = time.time()
            train_stats = self.train_epoch(epoch)
            val_stats = self.validate(epoch)
            self.scheduler.step()

            print(f"[Epoch {epoch:03d}] "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e} "
                  f"loss={train_stats['loss']:.4f} "
                  f"val_loss={val_stats['val_loss']:.4f} "
                  f"val_iou={val_stats['val_iou']:.4f} "
                  f"val_f1={val_stats['val_f1']:.4f} "
                  f"val_ber={val_stats['val_ber']:.4f} "
                  f"time={(time.time()-t0):.1f}s")
        
            # if self.vis_dir and (epoch % self.vis_every == 0):
            print(self.vis_dir, self.vis_every)
            self.save_predictions(epoch)

            if not math.isnan(val_stats["val_f1"]) and val_stats["val_f1"] > best_f1:
                best_f1 = val_stats["val_f1"]
                self.save_checkpoint(ckpt_path)
                print(f"[Epoch {epoch:03d}] New best F1={best_f1:.4f}. Saved: {ckpt_path}")

    def save_checkpoint(self, path: str) -> None:
        state = {
            "image_encoder": self.model.image_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(state, path)


def build_dataloaders(
    images_dir: str,
    masks_dir: str,
    size: int,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int
) -> Tuple[DataLoader, Optional[DataLoader]]:
    ds = SBUShadowDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        size=size,
        photometric_aug=True,
        seed=seed
    )
    if val_split <= 0.0:
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=sbu_collate), None
    val_len = int(len(ds) * val_split)
    train_len = len(ds) - val_len
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=sbu_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=sbu_collate)
    return train_loader, val_loader


    
def load_mobilesam_vit_t(ckpt_path: str | None, device: str = "cuda") -> nn.Module:
    model = sam_model_registry["vit_t"](checkpoint=ckpt_path)
    model.to(device)
    # freeze decoders, train only image encoder
    for p in model.prompt_encoder.parameters():
        p.requires_grad = False
    for p in model.mask_decoder.parameters():
        p.requires_grad = False
    for p in model.image_encoder.parameters():
        p.requires_grad = True
    model.train()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--masks_dir", type=str, required=True)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--ckpt_out", type=str, default="mobilesam_shadow_best.pt")
    p.add_argument("--pretrained", type=str, default="")  # optional MobileSAM checkpoint
    p.add_argument("--vis_dir", type=str, default="predictions")
    p.add_argument("--vis_every", type=int, default=1)
    p.add_argument("--vis_num", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model = load_mobilesam_vit_t(args.pretrained, device=args.device)
    train_loader, val_loader = build_dataloaders(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        size=args.size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        grad_clip=1.0,
        amp=args.amp,
        device=args.device,
        vis_dir=args.vis_dir,
        vis_every=args.vis_every,
        vis_num=10,
    )
    trainer.fit(args.ckpt_out)


if __name__ == "__main__":
    main()
