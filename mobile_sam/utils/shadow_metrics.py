import torch
from typing import Tuple


def _shadow_masks(
    probs: torch.Tensor,
    gt_mask: torch.Tensor,
    obj_mask: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (pred_shadow, gt_shadow) boolean masks with objects removed."""
    pred = probs >= threshold
    non_obj = ~obj_mask.bool()
    pred_shadow = pred.bool() & non_obj
    gt_shadow = gt_mask.bool() & non_obj
    return pred_shadow, gt_shadow


def compute_shadow_tp_union(
    probs: torch.Tensor,
    gt_mask: torch.Tensor,
    obj_mask: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Returns (TP, Union) for shadow IoU.
    Assumes `probs` are shadow probabilities. If you pass logits, apply sigmoid first.
    Shapes: (N, 1, H, W) or (N, H, W). Values in {0,1} after thresholding.
    """
    pred_shadow, gt_shadow = _shadow_masks(probs, gt_mask, obj_mask, threshold)
    tp = (pred_shadow & gt_shadow).sum().item()
    union = (pred_shadow | gt_shadow).sum().item()
    return float(tp), float(union)


def compute_shadow_iou(
    probs: torch.Tensor,
    gt_mask: torch.Tensor,
    obj_mask: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean shadow IoU over batch. Returns a scalar tensor."""
    pred_shadow, gt_shadow = _shadow_masks(probs, gt_mask, obj_mask, threshold)
    inter = (pred_shadow & gt_shadow).sum(dim=tuple(range(1, pred_shadow.ndim))).float()
    union = (pred_shadow | gt_shadow).sum(dim=tuple(range(1, pred_shadow.ndim))).float()
    iou = inter / (union + eps)
    return iou.mean()
