from __future__ import annotations

import argparse
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn

from mobile_sam.build_sam import sam_model_registry


def sam_normalize(img_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(img_uint8).to(device=device, dtype=torch.float32)  # H,W,3
    x = x.permute(2, 0, 1)  # 3,H,W
    mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device=device).view(3, 1, 1)
    return (x - mean) / std


def resize_pad_to_square(
    img: Image.Image,
    size: int = 1024,
) -> tuple[Image.Image, float, int, int]:
    w, h = img.size
    scale = size / max(w, h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    img_resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img_resized, (0, 0))
    return canvas, scale, nw, nh


def map_point_to_model(x: float, y: float, scale: float) -> tuple[float, float]:
    return x * scale, y * scale


def mask_to_original(
    prob_mask: np.ndarray,
    nw: int,
    nh: int,
    orig_w: int,
    orig_h: int,
    thr: float,
) -> np.ndarray:
    low = prob_mask[:nh, :nw]
    low_img = Image.fromarray((low * 255.0).astype(np.uint8))
    up = low_img.resize((orig_w, orig_h), Image.BILINEAR)
    up_np = np.asarray(up).astype(np.float32) / 255.0
    return (up_np >= thr).astype(np.uint8)


def overlay_mask(img_uint8: np.ndarray, bin_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = np.zeros_like(img_uint8, dtype=np.uint8)
    overlay[bin_mask.astype(bool)] = (255, 0, 0)
    mixed = img_uint8.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha
    return mixed.astype(np.uint8)


def draw_point(img_uint8: np.ndarray, x: float, y: float, color=(0, 255, 0)) -> np.ndarray:
    pil = Image.fromarray(img_uint8.copy())
    draw = ImageDraw.Draw(pil)
    r = 6
    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=color, width=3)
    draw.line([(x - r, y), (x + r, y)], fill=color, width=2)
    draw.line([(x, y - r), (x, y + r)], fill=color, width=2)
    return np.array(pil, dtype=np.uint8)


def _extract_image_encoder_state(state: Any) -> Any:
    if isinstance(state, dict) and "image_encoder" in state:
        return state["image_encoder"]
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def load_mobilesam_vit_t(ckpt_path: str | None, device: torch.device) -> nn.Module:
    model = sam_model_registry["vit_t"](checkpoint=None)
    model.to(device)
    model.requires_grad_(False)
    model.eval()

    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        state_dict = _extract_image_encoder_state(state)
        model.image_encoder.load_state_dict(state_dict, strict=False)

    return model


@torch.no_grad()
def forward_mobile_sam_single(
    model: nn.Module,
    image_1024_norm: torch.Tensor,  # [3,1024,1024]
    point_xy_1024: tuple[float, float],  # (x,y) in 1024 grid
    device: torch.device,
    multimask_output: bool = False,
) -> torch.Tensor:
    """Runs MobileSAM for a single point prompt and returns probabilities on a 1024x1024 grid."""
    x = image_1024_norm.unsqueeze(0)  # [1,3,1024,1024]
    image_embeddings = model.image_encoder(x)  # [1,256,64,64]
    image_pe = model.prompt_encoder.get_dense_pe()

    px, py = point_xy_1024
    coords = torch.tensor([[[px, py]]], device=device, dtype=torch.float32)  # [1,1,2]
    labels = torch.tensor([[1]], device=device, dtype=torch.int64)  # [1,1]
    sparse_embeds, dense_embeds = model.prompt_encoder(
        points=(coords, labels), boxes=None, masks=None
    )

    lowres_logits, iou_pred = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_embeds,
        dense_prompt_embeddings=dense_embeds,
        multimask_output=multimask_output,
    )  # [1,K,256,256], [1,K]

    if lowres_logits.shape[1] > 1:
        best_idx = torch.argmax(iou_pred, dim=1)
        chosen = lowres_logits[torch.arange(1, device=device), best_idx]  # [1,256,256]
    else:
        chosen = lowres_logits[:, 0]  # [1,256,256]

    logits = F.interpolate(
        chosen.unsqueeze(1),
        size=(1024, 1024),
        mode="bilinear",
        align_corners=False,
    )  # [1,1,1024,1024]
    probs = torch.sigmoid(logits)[0, 0]  # [1024,1024]
    return probs


class InteractiveSegmenter:
    def __init__(
        self,
        image_path: str,
        ckpt: str | None,
        device: str,
        thr: float,
        save_path: str | None,
    ) -> None:
        self.device = torch.device(device)
        self.model = load_mobilesam_vit_t(ckpt, self.device)
        self.thr = float(thr)
        self.save_path = save_path

        self.img_orig_pil = Image.open(image_path).convert("RGB")
        self.orig_w, self.orig_h = self.img_orig_pil.size
        self.img_orig_np = np.asarray(self.img_orig_pil)

        self.img_1024_padded, self.scale, self.nw, self.nh = resize_pad_to_square(
            self.img_orig_pil, 1024
        )
        self.img_1024_np = np.asarray(self.img_1024_padded)
        self.img_1024_norm = sam_normalize(self.img_1024_np, self.device)

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img_orig_np)
        self.ax.set_title("Click a point to segment (positive)")
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)

    def onclick(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        mx, my = map_point_to_model(x, y, self.scale)
        probs_1024 = forward_mobile_sam_single(
            self.model, self.img_1024_norm, (mx, my), device=self.device, multimask_output=False
        )
        prob_np = probs_1024.detach().cpu().numpy()  # 1024x1024
        bin_mask = mask_to_original(prob_np, self.nw, self.nh, self.orig_w, self.orig_h, self.thr)

        over = overlay_mask(self.img_orig_np, bin_mask, alpha=0.45)
        over = draw_point(over, x, y, color=(0, 255, 0))

        self.ax.clear()
        self.ax.imshow(over)
        self.ax.set_title(f"Point=({int(x)},{int(y)}) | thr={self.thr}")
        self.fig.canvas.draw_idle()

        if self.save_path:
            Image.fromarray(over).save(self.save_path)
            print(f"[INFO] Saved: {self.save_path}")

    def run(self) -> None:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--save", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = InteractiveSegmenter(
        image_path=args.image, ckpt=args.ckpt, device=args.device, thr=args.thr, save_path=args.save
    )
    app.run()


if __name__ == "__main__":
    main()
