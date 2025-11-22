from typing import Dict, List, Optional, Tuple
import os
import math
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
from typing import Any
import wandb
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

# from mobile_sam.utils.sbu_dataset import SBUShadowDataset, sbu_collate
from mobile_sam.utils.obj_prompt_shadow_dataset import ObjPromptShadowDataset, two_mask_collate
from mobile_sam.build_sam import sam_model_registry
from mobile_sam.utils.common import sam_denormalize, color_overlay, make_panel, make_panel_with_points

from mobile_sam.prune import apply_pruning, remove_pruning_reparam


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
def compute_metrics(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5, obj_mask: torch.Tensor = None) -> Dict[str, float]:
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
    metrics = {"iou": iou, "f1": f1, "ber": ber}
    if obj_mask is not None:
        shadow_pred = (pred - obj_mask).clamp(min=0)
        shadow_gt = (target - obj_mask).clamp(min=0)
        intersection = (shadow_pred * shadow_gt).sum().item()
        union = ((shadow_pred + shadow_gt) > 0).float().sum().item()
        shadow_iou = intersection / max(1.0, union)
        metrics["shadow_iou"] = shadow_iou
    return metrics


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

    for i in range(B):
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
        wandb_run: Optional[Any] = None,
        wandb_images: int = 16
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
            self.scaler = torch.amp.GradScaler(self.device, enabled=enable_amp)
        except TypeError:
            self.scaler = torch.amp.GradScaler(enabled=enable_amp)
        self.vis_dir = vis_dir
        self.vis_every = vis_every
        self.vis_num = vis_num
        self.wandb = wandb_run
        self.wandb_images = wandb_images

    
    def save_predictions(self, epoch: int) -> None:
        if not self.vis_dir:
            return
        out_dir = os.path.join(self.vis_dir, f"epoch_{epoch:03d}")
        os.makedirs(out_dir, exist_ok=True)

        loader = self.val_loader if self.val_loader is not None else self.train_loader
        self.model.eval()

        saved = 0
        wb_imgs: List[Any] = []

        with torch.no_grad():
            for b_idx, batch in enumerate(loader):
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                points = [p.to(self.device) for p in batch["points"]]
                labels = [l.to(self.device) for l in batch["point_labels"]]
                boxes = [b.to(self.device) for b in batch["boxes"]]

                logits = forward_mobile_sam(self.model, images, points, labels, boxes)
                preds = torch.sigmoid(logits) >= 0.5

                for i in range(images.size(0)):
                    img_np = sam_denormalize(images[i])
                    gt_np = masks[i, 0].cpu().numpy() > 0.5
                    pred_np = preds[i, 0].cpu().numpy() > 0.5

                    # Include bounding boxes in the panel
                    box = boxes[i]
                    if box.numel() == 4:
                        x0, y0, x1, y1 = box.cpu().numpy()
                        panel = make_panel_with_points(img_np, gt_np, pred_np, points[i].cpu().numpy())
                        draw = ImageDraw.Draw(panel)
                        draw.rectangle([x0, y0, x1, y1], outline="blue", width=2)
                    else:
                        panel = make_panel_with_points(img_np, gt_np, pred_np, points[i].cpu().numpy())

                    # panel_path = os.path.join(out_dir, f"{saved:03d}.png")
                    # panel.save(panel_path)

                    if self.wandb and saved < self.wandb_images:
                        wb_imgs.append(wandb.Image(panel, caption=f"Sample {saved}"))

                    saved += 1

        if self.wandb and wb_imgs:
            self.wandb.log({"predictions": wb_imgs}, step=epoch)

        print(f"[Vis] Saved {saved} images to {out_dir}")
        self.model.train()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        agg = {"iou": 0.0, "f1": 0.0, "ber": 0.0, "shadow_iou": 0.0}
        for batch in tqdm(self.train_loader):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            obj_mask = batch["obj_mask"].to(self.device, non_blocking=True) if "obj_mask" in batch else None
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
            m = compute_metrics(logits, masks, thr=0.5, obj_mask=obj_mask) if obj_mask is not None else compute_metrics(logits, masks, thr=0.5)
            for k in agg:
                if k in m:
                    agg[k] += m[k]

        for k in agg: agg[k] /= max(1, n_batches)
        return {"train_loss": total_loss / max(1, n_batches), "train_iou": agg["iou"], "train_f1": agg["f1"], "train_ber": agg["ber"], "train_shadow_iou": agg["shadow_iou"]}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        if self.val_loader is None:
            return {"val_loss": math.nan, "val_iou": math.nan, "val_f1": math.nan, "val_ber": math.nan}
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        agg = {"iou": 0.0, "f1": 0.0, "ber": 0.0, "shadow_iou": 0.0}
        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            points = [p.to(self.device) for p in batch["points"]]
            labels = [l.to(self.device) for l in batch["point_labels"]]
            boxes = [b.to(self.device) for b in batch["boxes"]]
            obj_mask = batch["obj_mask"].to(self.device, non_blocking=True) if "obj_mask" in batch else None
            logits = forward_mobile_sam(self.model, images, points, labels, boxes)
            loss = self.loss_fn(logits, masks)
            total_loss += float(loss.item()); n_batches += 1
            m = compute_metrics(logits, masks, thr=0.5, obj_mask=obj_mask) if obj_mask is not None else compute_metrics(logits, masks, thr=0.5)
            for k in agg:
                if k in m:
                    agg[k] += m[k]
        for k in agg: agg[k] /= max(1, n_batches)
        return {"val_loss": total_loss / max(1, n_batches), "val_iou": agg["iou"], "val_f1": agg["f1"], "val_ber": agg["ber"], "val_shadow_iou": agg["shadow_iou"]}

    def fit(self, ckpt_path: str) -> None:
        best_f1 = -1.0
        ckpt_last = ckpt_path.replace('.pt', '_last.pt') if ckpt_path.endswith('.pt') else ckpt_path + '_last.pt'
        ckpt_best = ckpt_path.replace('.pt', '_best.pt') if ckpt_path.endswith('.pt') else ckpt_path + '_best.pt'
        for epoch in tqdm(range(1, self.max_epochs + 1)):
            t0 = time.time()
            train_stats = self.train_epoch(epoch)
            val_stats = self.validate(epoch)
            self.scheduler.step()


            print(f"[Epoch {epoch:03d}] "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e} "
                f"train_loss={train_stats['train_loss']:.4f} "
                f"train_iou={train_stats['train_iou']:.4f} "
                f"train_f1={train_stats['train_f1']:.4f} "
                f"train_ber={train_stats['train_ber']:.4f} "
                f"train_shadow_iou={train_stats['train_shadow_iou']:.4f} "
                f"val_loss={val_stats['val_loss']:.4f} "
                f"val_iou={val_stats['val_iou']:.4f} "
                f"val_f1={val_stats['val_f1']:.4f} "
                f"val_ber={val_stats['val_ber']:.4f} "
                f"val_shadow_iou={val_stats.get('val_shadow_iou', float('nan')):.4f} "
                f"time={(time.time()-t0):.1f}s")

            if self.wandb:
                log_dict = {
                    "epoch": epoch,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "train/loss": train_stats["train_loss"],
                    "train/iou": train_stats["train_iou"],
                    "train/f1": train_stats["train_f1"],
                    "train/ber": train_stats["train_ber"],
                    "val/loss": val_stats["val_loss"],
                    "val/iou": val_stats["val_iou"],
                    "val/f1": val_stats["val_f1"],
                    "val/ber": val_stats["val_ber"],
                }
                if "val_shadow_iou" in val_stats:
                    log_dict["train/shadow_iou"] = train_stats["train_shadow_iou"]
                    log_dict["val/shadow_iou"] = val_stats["val_shadow_iou"]
                self.wandb.log(log_dict, step=epoch)

                print(self.vis_dir, self.vis_every)
                self.save_predictions(epoch)

                self.save_checkpoint(ckpt_last)

                if not math.isnan(val_stats["val_f1"]) and val_stats["val_f1"] > best_f1:
                    best_f1 = val_stats["val_f1"]
                    self.save_checkpoint(ckpt_best)
                    print(f"[Epoch {epoch:03d}] New best F1={best_f1:.4f}. Saved: {ckpt_best}")

    def save_checkpoint(self, path: str) -> None:
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(state, path)


def build_dataloaders(
    images_dir: str,
    masks_dir: str,
    object_masks_dir: str,
    size: int,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int
) -> Tuple[DataLoader, Optional[DataLoader]]:
    ds = ObjPromptShadowDataset(
        images_dir=images_dir,
        obj_masks_dir=object_masks_dir,
        target_masks_dir=masks_dir,
        size=size,
        seed=seed,
        return_obj_mask=True,
    )
    if val_split <= 0.0:
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=two_mask_collate), None
    val_len = int(len(ds) * val_split)
    train_len = len(ds) - val_len
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=two_mask_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=two_mask_collate)
    return train_loader, val_loader


    
def load_mobilesam_vit_t(ckpt_path: str | None, device: str = "cuda") -> nn.Module:
    model = sam_model_registry["vit_t"](checkpoint=ckpt_path)
    model.to(device)
    for p in model.prompt_encoder.parameters():
        p.requires_grad = False
    for p in model.mask_decoder.parameters():
        p.requires_grad = False
    for p in model.image_encoder.parameters():
        p.requires_grad = True
    model.train()
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    set_seed(cfg.system.seed)

    base_images_dir = cfg.data.images_dir
    base_obj_masks_dir = cfg.data.obj_masks_dir
    base_tgt_masks_dir = cfg.data.masks_dir

    augmentations = [
        ("flip", "flip"),
        # ("blur", "blur"),
        # ("rotate", "rotate"),
        # ("hue", "hue"),
        # ("bright", "bright"),
    ]

    for aug_name, aug_folder in augmentations:
        print(f"\n[INFO] Training: base + {aug_name}")

        images_dir = [base_images_dir, f"{base_images_dir}_aug/{aug_folder}"]
        obj_masks_dir = [base_obj_masks_dir, f"{base_obj_masks_dir}_aug/{aug_folder}"]
        target_masks_dir = [base_tgt_masks_dir, f"{base_tgt_masks_dir}_aug/{aug_folder}"]

        model = load_mobilesam_vit_t(None, device=cfg.system.device)
        train_loader, val_loader = build_dataloaders(
            images_dir=images_dir,
            masks_dir=target_masks_dir,
            object_masks_dir=obj_masks_dir,
            size=cfg.data.size,
            batch_size=cfg.data.batch_size,
            val_split=cfg.train.val_split,
            num_workers=cfg.system.num_workers,
            seed=cfg.system.seed
        )
        first_batch = next(iter(train_loader))
        print(f"[INFO] Train dataset size: {len(train_loader.dataset)} samples")
        print(first_batch.keys())

        wandb_run = None
        if cfg.wandb.enabled:
            config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

            config_dict["current_augmentation"] = aug_name

            config_dict["pruning_type"] = cfg.model.pruning.mode
            config_dict["pruning_amount"] = cfg.model.pruning.amount

            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                mode=cfg.wandb.mode,
                config=config_dict
            )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            max_epochs=cfg.train.epochs,
            grad_clip=1.0,
            amp=cfg.train.amp,
            device=cfg.system.device,
            vis_dir=cfg.test.vis_dir,
            vis_every=cfg.test.vis_every,
            vis_num=cfg.test.vis_num,
            wandb_run=wandb_run,
            wandb_images=cfg.wandb.wandb_images_num,
        )

        if cfg.train.resume_ckpt:
            print(f"[INFO] Loading checkpoint from {cfg.train.resume_ckpt}")
            checkpoint = torch.load(cfg.train.resume_ckpt, map_location=cfg.system.device)
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"])
                elif all(isinstance(k, str) and k.startswith(("image_encoder.", "prompt_encoder.", "mask_decoder.")) for k in checkpoint.keys()):
                    model.load_state_dict(checkpoint)
                elif all(k in checkpoint for k in ("image_encoder", "prompt_encoder", "mask_decoder")):
                    flat_state = {}
                    for submodule in ("image_encoder", "prompt_encoder", "mask_decoder"):
                        subdict = checkpoint[submodule]
                        for k, v in subdict.items():
                            flat_state[f"{submodule}.{k}"] = v
                    model.load_state_dict(flat_state, strict=False)
                    print("[INFO] Loaded checkpoint (image_encoder, prompt_encoder, mask_decoder)")
                else:
                    print("[ERROR] Checkpoint does not contain 'model' key or does not look like a model state_dict. Keys:", list(checkpoint.keys()))
                    raise RuntimeError("Invalid checkpoint format!")
            else:
                print("[ERROR] Checkpoint is not a dictionary!")
                raise RuntimeError("Invalid checkpoint format!")
            try:
                if isinstance(checkpoint, dict):
                    if "optimizer" in checkpoint:
                        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                    if "scheduler" in checkpoint:
                        trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                    print("[INFO] Optimizer and scheduler state loaded.")
            except Exception as e:
                print(f"[WARN] Could not load optimizer/scheduler state: {e}")
        elif cfg.model.pretrained_path:
            print(f"[INFO] Loading pretrained weights from {cfg.model.pretrained_path}")
            state = torch.load(cfg.model.pretrained_path, map_location=cfg.system.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)

        if cfg.model.pruning.enabled:
            print(f"[INFO] Pruning ENABLED. Mode: {cfg.model.pruning.mode}, Amount: {cfg.model.pruning.amount}")
            apply_pruning(
                model=trainer.model,
                mode=cfg.model.pruning.mode,
                amount=cfg.model.pruning.amount,
                include_linear=True,
                structured_n=1,
                structured_dim=0,
            )
        else:
            print("[INFO] Pruning DISABLED.")

        trainer.fit(cfg.train.output_ckpt)

        if cfg.model.pruning.enabled:
            remove_pruning_reparam(trainer.model)
            suffix = "_pruned"
        else:
            suffix = ""

        ckpt_final = cfg.train.output_ckpt.replace('.pt', f'_{aug_name}_final{suffix}.pt') if cfg.train.output_ckpt.endswith('.pt') else cfg.train.output_ckpt + f'_{aug_name}_final{suffix}.pt'
        torch.save(trainer.model.state_dict(), ckpt_final)
        print(f"[INFO] Saved final pruned model to {ckpt_final}")

        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()
