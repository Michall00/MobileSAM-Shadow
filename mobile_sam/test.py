from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import argparse
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
import wandb
from dotenv import load_dotenv

from mobile_sam.train import build_dataloaders, load_mobilesam_vit_t, set_seed
from mobile_sam.utils.sbu_dataset import sbu_collate
from mobile_sam.utils.obj_prompt_shadow_dataset import ObjPromptShadowDataset
from mobile_sam.build_sam import sam_model_registry
from mobile_sam.utils.common import sam_denormalize, color_overlay, make_panel
from mobile_sam.utils.shadow_metrics import compute_shadow_tp_union, compute_shadow_iou


def compute_counts(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).float()
    tp = (pred * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


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
    metrics = {"iou": iou, "f1": f1, "ber": ber, "precision": prec, "recall": rec}
    if obj_mask is not None:
        shadow_pred = (pred - obj_mask).clamp(min=0)
        shadow_gt = (target - obj_mask).clamp(min=0)
        intersection = (shadow_pred * shadow_gt).sum().item()
        union = ((shadow_pred + shadow_gt) > 0).float().sum().item()
        shadow_iou = intersection / max(1.0, union)
        metrics["shadow_iou"] = shadow_iou
    return metrics


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


@torch.no_grad()
def forward_mobile_sam(
    model: torch.nn.Module,
    images: torch.Tensor,
    points: List[torch.Tensor],
    point_labels: List[torch.Tensor],
    boxes: List[torch.Tensor],
    multimask_output: bool = False
) -> torch.Tensor:
    device = images.device
    B, _, H, W = images.shape
    assert H == 1024 and W == 1024

    image_embeddings = model.image_encoder(images)
    image_pe = model.prompt_encoder.get_dense_pe()

    logits_out: List[torch.Tensor] = []
    for i in range(B):
        pts_i = points[i]
        lbs_i = point_labels[i]
        box_i = boxes[i]

        pts_tuple = None
        if pts_i.numel() > 0:
            coords = pts_i.to(device=device, dtype=torch.float32).unsqueeze(0)
            labels = lbs_i.to(device=device, dtype=torch.int64).unsqueeze(0)
            pts_tuple = (coords, labels)

        box_tensor = None
        if box_i.numel() == 4:
            box_tensor = box_i.to(device=device, dtype=torch.float32).unsqueeze(0)

        sparse_embeds, dense_embeds = model.prompt_encoder(
            points=pts_tuple, boxes=box_tensor, masks=None
        )

        lowres_logits, iou_pred = model.mask_decoder(
            image_embeddings=image_embeddings[i:i+1],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeds,
            dense_prompt_embeddings=dense_embeds,
            multimask_output=multimask_output
        )

        if lowres_logits.shape[1] > 1:
            best_idx = torch.argmax(iou_pred, dim=1)
            chosen = lowres_logits[torch.arange(1, device=device), best_idx]
        else:
            chosen = lowres_logits[:, 0]

        chosen = F.interpolate(chosen.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)
        logits_out.append(chosen)

    return torch.cat(logits_out, dim=0)


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
    print("Saving predictions to", vis_dir)

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

    print(f"[Test] Saved {saved} images to {vis_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--masks_dir", type=str, required=True)
    p.add_argument("--obj_masks_dir", type=str, required=True)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--pretrained", type=str, default="")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--vis_dir", type=str, default="predictions_val_shadow")
    p.add_argument("--vis_num", type=int, default=50)
    p.add_argument("--save_csv", type=str, default="test_metrics.csv")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="mobilesam-shadow")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--wandb_images", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    set_seed(args.seed)

    model = load_mobilesam_vit_t(None, device=args.device)

    if args.ckpt:
        print(f"[INFO] Loading checkpoint from {args.ckpt}")
        state = torch.load(args.ckpt, map_location=args.device)
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
        else:
            print("[ERROR] Checkpoint format not recognized.")
            return
    elif args.pretrained:
        print(f"[INFO] Loading pretrained weights from {args.pretrained}")
        state = torch.load(args.pretrained, map_location=args.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    loader, _ = build_dataloaders(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        object_masks_dir=args.obj_masks_dir,
        size=args.size,
        batch_size=args.batch_size,
        val_split=0.0,  # No validation split for testing
        num_workers=args.num_workers,
        seed=args.seed
    )

    wb = None
    if args.wandb:
        wb = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            mode=args.wandb_mode,
            config=vars(args),
        )

    total_tp = total_tn = total_fp = total_fn = 0.0
    shadow_tp_total, shadow_union_total = 0.0, 0.0
    saved = 0

    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["batch_idx", "precision", "recall", "f1", "iou", "ber", "shadow_iou"])

            with torch.no_grad():
                for b_idx, batch in enumerate(tqdm(loader)):
                    images = batch["image"].to(args.device, non_blocking=True)
                    masks = batch["mask"].to(args.device, non_blocking=True)
                    points = [p.to(args.device) for p in batch["points"]]
                    labels = [l.to(args.device) for l in batch["point_labels"]]
                    boxes = [b.to(args.device) for b in batch["boxes"]]
                    obj_mask = batch["obj_mask"].to(args.device, non_blocking=True) if "obj_mask" in batch else None

                    with torch.amp.autocast(args.device, enabled=(args.amp and args.device == "cuda")):
                        logits = forward_mobile_sam(model, images, points, labels, boxes)
                        probs = torch.sigmoid(logits)

                    counts = compute_counts(logits, masks, thr=args.thr)
                    total_tp += counts["tp"]; total_tn += counts["tn"]; total_fp += counts["fp"]; total_fn += counts["fn"]

                    shadow_tp, shadow_union = None, None
                    shadow_iou = None
                    if obj_mask is not None:
                        shadow_tp, shadow_union = compute_shadow_tp_union(probs, masks, obj_mask, args.thr)
                        shadow_iou = compute_shadow_iou(probs, masks, obj_mask, args.thr).item()
                        shadow_tp_total += shadow_tp
                        shadow_union_total += shadow_union

                    m = finalize_metrics(counts["tp"], counts["tn"], counts["fp"], counts["fn"], shadow_tp, shadow_union)
                    m["shadow_iou"] = shadow_iou if shadow_iou is not None else float("nan")
                    writer.writerow([b_idx, m["precision"], m["recall"], m["f1"], m["iou"], m["ber"], m.get("shadow_iou", float("nan"))])

                    if args.vis_dir and (args.vis_num < 0 or saved < args.vis_num):
                        saved = save_panels_and_log(
                            images, masks, probs, args.vis_dir, b_idx, wb, args.wandb_images, saved, boxes
                        )

    global_metrics = finalize_metrics(total_tp, total_tn, total_fp, total_fn, shadow_tp_total, shadow_union_total)
    print(
        f"[TEST] IoU={global_metrics['iou']:.4f} "
        f"F1={global_metrics['f1']:.4f} "
        f"BER={global_metrics['ber']:.4f} "
        f"Precision={global_metrics['precision']:.4f} "
        f"Recall={global_metrics['recall']:.4f} "
        f"Shadow IoU={global_metrics.get('shadow_iou', float('nan')):.4f}"
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
        save_predictions(model, loader, args.device, args.vis_dir, args.vis_num, wb, args.wandb_images)

        wb.finish()


if __name__ == "__main__":
    main()
