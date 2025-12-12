from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F


def sam_denormalize_float(img_t: torch.Tensor) -> np.ndarray:
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


def make_panel_with_points(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, points: np.ndarray) -> Image.Image:
    panel = make_panel(img, gt, pred)
    w, h = img.shape[1], img.shape[0]
    draw = ImageDraw.Draw(panel)

    for px, py in points:
        for offset in [0, w, 2 * w]:
            x, y = px + offset, py
            r = 7
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 0), outline=(0, 0, 0), width=2)
    return panel
