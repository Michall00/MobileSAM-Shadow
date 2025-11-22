from __future__ import annotations
import os
import csv
import logging
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from PIL import ImageDraw
from rich.logging import RichHandler
from torch.utils.data import DataLoader
from tqdm import tqdm

from mobile_sam.train import build_dataloaders, load_mobilesam_vit_t, set_seed
from mobile_sam.utils.common import make_panel, sam_denormalize
from mobile_sam.utils.eval import forward_mobile_sam
from mobile_sam.utils.shadow_metrics import compute_shadow_iou, compute_shadow_tp_union


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False
        )
    ]
)

log = logging.getLogger(__name__)

def compute_counts(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).float()
    tp = (pred * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def finalize_metrics(tp: float, tn: float, fp: float, fn: float, shadow_tp: Optional[float] = None, shadow_union: Optional[float] = None) -> Dict[str, float]:
    iou = tp / max(1.0, (tp + fp + fn))
    prec = tp / max(1.0, (tp + fp))
    rec = tp / max(1.0, (tp + fn))
    f1 = 2 * prec * rec / max(1e-6, (prec + rec))
    ber = 0.5 * (fn / max(1.0, tp + fn) + fp / max(1.0, tn + fp))
    metrics = {"iou": iou, "f1": f1, "ber": ber, "precision": prec, "recall": rec}

    if shadow_tp is not None and shadow_union is not None:
        shadow_iou = shadow_tp / max(1.0, shadow_union)
        metrics["shadow_iou"] = shadow_iou

    return metrics


def save_panels_and_log(
    images: torch.Tensor,
    masks: torch.Tensor,
    probs: torch.Tensor,
    out_dir: str,
    batch_idx: int,
    wb: Optional[wandb.sdk.wandb_run.Run],
    wb_limit: int,
    saved_total: int,
    boxes: List[torch.Tensor]
) -> int:
    os.makedirs(out_dir, exist_ok=True)
    wb_imgs: List[Any] = []
    B = images.size(0)
    for i in range(B):
        img_np = sam_denormalize(images[i])
        gt = (masks[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        pr = (probs[i, 0].detach().cpu().numpy() >= 0.5).astype(np.uint8)
        panel = make_panel(img_np, gt, pr)

        # Draw bounding boxes
        box = boxes[i]
        if box.numel() == 4:
            x0, y0, x1, y1 = box.cpu().numpy()
            draw = ImageDraw.Draw(panel)
            draw.rectangle([x0, y0, x1, y1], outline="blue", width=2)

        out_path = os.path.join(out_dir, f"b{batch_idx:05d}_i{i:02d}.png")
        panel.save(out_path)
        saved_total += 1
        if wb and (wb_limit < 0 or len(wb_imgs) < wb_limit):
            wb_imgs.append(wandb.Image(panel, caption=f"b{batch_idx:05d}_i{i:02d}"))
    if wb and wb_imgs:
        wb.log({"test/predictions": wb_imgs})
    return saved_total


def save_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    vis_dir: str,
    vis_num: int,
    wandb_run: Optional[wandb.sdk.wandb_run.Run],
    wandb_images: int
) -> None:
    if not vis_dir:
        return
    os.makedirs(vis_dir, exist_ok=True)
    log.info(f"Saving predictions to {vis_dir}")

    model.eval()
    saved = 0
    wb_imgs: List[Any] = []

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            points = [p.to(device) for p in batch["points"]]
            labels = [l.to(device) for l in batch["point_labels"]]
            boxes = [b.to(device) for b in batch["boxes"]]

            logits = forward_mobile_sam(model, images, points, labels, boxes)
            probs = torch.sigmoid(logits)

            for i in tqdm(range(images.size(0))):
                img_np = sam_denormalize(images[i])
                gt_np = masks[i, 0].cpu().numpy() > 0.5
                pred_np = probs[i, 0].cpu().numpy() > 0.5

                # Include bounding boxes in the panel
                box = boxes[i]
                if box.numel() == 4:
                    x0, y0, x1, y1 = box.cpu().numpy()
                    panel = make_panel(img_np, gt_np, pred_np)
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle([x0, y0, x1, y1], outline="blue", width=2)
                else:
                    panel = make_panel(img_np, gt_np, pred_np)

                out_path = os.path.join(vis_dir, f"b{b_idx:05d}_i{i:02d}.png")
                panel.save(out_path)
                saved += 1

                if saved >= vis_num:
                    break

                if wandb_run and saved < wandb_images:
                    wb_imgs.append(wandb.Image(panel, caption=f"b{b_idx:05d}_i{i:02d}"))

    if wandb_run and wb_imgs:
        wandb_run.log({"test/predictions": wb_imgs})

    log.info(f"[Test] Saved {saved} images to {vis_dir}")


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    set_seed(cfg.system.seed)

    device = cfg.system.device
    amp_enabled = getattr(cfg.train, "amp", True)
    thr = getattr(cfg.test, "thr", 0.5)

    model = load_mobilesam_vit_t(None, device=device)

    if getattr(cfg.test, "ckpt_path", None):
        ckpt = cfg.test.ckpt_path
        log.info(f"[INFO] Loading checkpoint from {ckpt}")
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
        else:
            log.error(f"[ERROR] Checkpoint format not recognized.")
            return
    elif getattr(cfg.model, "pretrained_path", None):
        pre = cfg.model.pretrained_path
        log.info(f"[INFO] Loading pretrained weights from {pre}")
        state = torch.load(pre, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    loader, _ = build_dataloaders(
        images_dir=cfg.data.images_dir,
        masks_dir=cfg.data.masks_dir,
        object_masks_dir=cfg.data.obj_masks_dir,
        size=cfg.data.size,
        batch_size=cfg.data.batch_size,
        val_split=0.0,  # No validation split for testing
        num_workers=cfg.system.num_workers,
        seed=cfg.system.seed,
    )

    wb = None
    if getattr(cfg, "wandb", None) and cfg.wandb.enabled:
        wb_cfg = OmegaConf.to_container(cfg, resolve=True)
        wb = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.get("run_name", None),
            mode=cfg.wandb.mode,
            config=wb_cfg,
        )

    total_tp = total_tn = total_fp = total_fn = 0.0
    shadow_tp_total, shadow_union_total = 0.0, 0.0
    saved = 0

    save_csv = getattr(cfg.test, "save_csv", None)

    if save_csv:
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["batch_idx", "precision", "recall", "f1", "iou", "ber", "shadow_iou"])

            with torch.no_grad():
                for b_idx, batch in enumerate(tqdm(loader)):
                    images = batch["image"].to(device, non_blocking=True)
                    masks = batch["mask"].to(device, non_blocking=True)
                    points = [p.to(device) for p in batch["points"]]
                    labels = [l.to(device) for l in batch["point_labels"]]
                    boxes = [b.to(device) for b in batch["boxes"]]
                    obj_mask = batch.get("obj_mask")
                    if obj_mask is not None:
                        obj_mask = obj_mask.to(device, non_blocking=True)

                    with torch.amp.autocast(device, enabled=(amp_enabled and device == "cuda")):
                        logits = forward_mobile_sam(model, images, points, labels, boxes)
                        probs = torch.sigmoid(logits)

                    counts = compute_counts(logits, masks, thr=thr)
                    total_tp += counts["tp"]; total_tn += counts["tn"]; total_fp += counts["fp"]; total_fn += counts["fn"]

                    shadow_tp = shadow_union = None
                    shadow_iou = None
                    if obj_mask is not None:
                        shadow_tp, shadow_union = compute_shadow_tp_union(probs, masks, obj_mask, thr)
                        shadow_iou = compute_shadow_iou(probs, masks, obj_mask, thr).item()
                        shadow_tp_total += shadow_tp
                        shadow_union_total += shadow_union

                    m = finalize_metrics(counts["tp"], counts["tn"], counts["fp"], counts["fn"], shadow_tp, shadow_union)
                    m["shadow_iou"] = shadow_iou if shadow_iou is not None else float("nan")
                    writer.writerow([b_idx, m["precision"], m["recall"], m["f1"], m["iou"], m["ber"], m.get("shadow_iou", float("nan"))])

                    vis_dir = getattr(cfg.test, "vis_dir", None)
                    vis_num = getattr(cfg.test, "vis_num", -1)
                    if vis_dir and (vis_num < 0 or saved < vis_num):
                        saved = save_panels_and_log(
                            images, masks, probs, vis_dir, b_idx, wb, cfg.wandb.wandb_images_num, saved, boxes
                        )

    global_metrics = finalize_metrics(total_tp, total_tn, total_fp, total_fn, shadow_tp_total, shadow_union_total)
    log.info(
        "[TEST] IoU=%.4f F1=%.4f BER=%.4f Precision=%.4f Recall=%.4f Shadow IoU=%.4f",
        global_metrics['iou'], global_metrics['f1'], global_metrics['ber'],
        global_metrics['precision'], global_metrics['recall'], global_metrics.get('shadow_iou', float('nan'))
    )

    if wb:
        wb.log({
            "test/iou": global_metrics["iou"],
            "test/f1": global_metrics["f1"],
            "test/ber": global_metrics["ber"],
            "test/precision": global_metrics["precision"],
            "test/recall": global_metrics["recall"],
            "test/shadow_iou": global_metrics.get("shadow_iou", float("nan"))
        })
        save_predictions(model, loader, device, getattr(cfg.test, "vis_dir", None), getattr(cfg.test, "vis_num", 0), wb, cfg.wandb.wandb_images_num)

        wb.finish()


if __name__ == "__main__":
    main()
