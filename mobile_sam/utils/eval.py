from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


def compute_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    thr: float = 0.5,
    obj_mask: Optional[torch.Tensor] = None,
    include_precision_recall: bool = False,
) -> Dict[str, float]:
    """Basic segmentation metrics with optional shadow IoU."""
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

    metrics: Dict[str, float] = {"iou": iou, "f1": f1, "ber": ber}

    if include_precision_recall:
        metrics["precision"] = prec
        metrics["recall"] = rec

    if obj_mask is not None:
        artifact_pred = (pred - obj_mask).clamp(min=0)
        artifact_gt = (target - obj_mask).clamp(min=0)
        
        intersection = (artifact_pred * artifact_gt).sum().item()

        union_pixels = ((artifact_pred + artifact_gt) > 0).float().sum().item()
        metrics["shadow_iou"] = intersection / max(1.0, union_pixels)
        
        area_pred = artifact_pred.sum().item()
        area_gt = artifact_gt.sum().item()
        metrics["shadow_f1"] = (2 * intersection) / max(1e-6, area_pred + area_gt)

    return metrics


def forward_mobile_sam(
    model: nn.Module,
    images: torch.Tensor,  # [B,3,1024,1024]
    points: List[torch.Tensor],
    point_labels: List[torch.Tensor],
    boxes: List[torch.Tensor],
    multimask_output: bool = False,
) -> torch.Tensor:
    """
    MobileSAM forward pass for a batch of prompts.
    Returns logits of shape [B,1,1024,1024].
    """
    device = images.device
    _, _, H, W = images.shape
    assert H == 1024 and W == 1024, "Expected 1024x1024 inputs."

    image_embeddings = model.image_encoder(images)
    image_pe = model.prompt_encoder.get_dense_pe()

    logits_out: List[torch.Tensor] = []
    for i, (pts_i, lbs_i, box_i) in enumerate(zip(points, point_labels, boxes)):
        pts_tuple = None
        if pts_i.numel() > 0:
            coords = pts_i.to(device=device, dtype=torch.float32).unsqueeze(0)  # [1,Ni,2]
            labels = lbs_i.to(device=device, dtype=torch.int64).unsqueeze(0)    # [1,Ni]
            pts_tuple = (coords, labels)

        box_tensor = None
        if box_i.numel() == 4:
            box_tensor = box_i.to(device=device, dtype=torch.float32).unsqueeze(0)  # [1,4]

        sparse_embeds, dense_embeds = model.prompt_encoder(points=pts_tuple, boxes=box_tensor, masks=None)

        lowres_logits, iou_pred = model.mask_decoder(
            image_embeddings=image_embeddings[i:i+1],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeds,
            dense_prompt_embeddings=dense_embeds,
            multimask_output=multimask_output,
        )

        if lowres_logits.shape[1] > 1:
            best_idx = torch.argmax(iou_pred, dim=1)
            chosen = lowres_logits[torch.arange(1, device=device), best_idx]
        else:
            chosen = lowres_logits[:, 0]

        upsampled = F.interpolate(
            chosen.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        logits_out.append(upsampled)

    return torch.cat(logits_out, dim=0)
