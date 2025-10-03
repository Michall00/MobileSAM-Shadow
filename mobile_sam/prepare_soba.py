import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw

try:
    from pycocotools import mask as coco_mask
except Exception:
    coco_mask = None


@dataclass
class Ann:
    id: int
    image_id: int
    category_id: int  # 1=Object, 2=Shadow
    association: Optional[int]
    rle: Optional[Dict]
    bbox: Tuple[float, float, float, float]


@dataclass
class Img:
    id: int
    file_name: str
    width: int
    height: int
    object_mask_path: Optional[str]
    shadow_mask_path: Optional[str]


def load_json(p: Path) -> Dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_bool_mask_from_rle(rle: Dict, h: int, w: int) -> np.ndarray:
    if coco_mask is None:
        raise RuntimeError("pycocotools is required to decode RLE masks")
    cnts = rle.get("counts")
    if isinstance(cnts, list):
        rle_norm = coco_mask.frPyObjects(rle, h, w)
    elif isinstance(cnts, (str, bytes)):
        rle_norm = rle
    else:
        rle_norm = coco_mask.frPyObjects(rle, h, w)
    m = coco_mask.decode(rle_norm)
    if m.ndim == 3:
        m = np.any(m, axis=2)
    return m.astype(np.uint8)



def load_mask_from_png(p: Path, shape: Tuple[int, int]) -> np.ndarray:
    arr = np.array(Image.open(p).convert("L"))
    if arr.shape != shape:
        arr = np.array(Image.fromarray(arr).resize((shape[1], shape[0]), Image.NEAREST))
    return (arr > 0).astype(np.uint8)


def sample_point(mask: np.ndarray, rng: random.Random) -> Tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("Empty object mask")
    i = rng.randrange(ys.size)
    return int(xs[i]), int(ys[i])


def save_mask_png(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)


def save_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).save(dst, quality=95)


def save_point_json(img_rel: str, point: Tuple[int, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"image": img_rel, "point": [point[0], point[1]]}, f, ensure_ascii=False)


def build_indices(meta: Dict) -> Tuple[Dict[int, Img], Dict[int, List[Ann]], Dict[int, Dict[int, List[Ann]]]]:
    images: Dict[int, Img] = {}
    for im in meta["images"]:
        images[im["id"]] = Img(
            id=im["id"],
            file_name=im["file_name"],
            width=int(im["width"]),
            height=int(im["height"]),
            object_mask_path=im.get("object_mask_path"),
            shadow_mask_path=im.get("shadow_mask_path"),
        )
    anns_by_image: Dict[int, List[Ann]] = {}
    for an in meta["annotations"]:
        ann = Ann(
            id=an["id"],
            image_id=an["image_id"],
            category_id=an["category_id"],
            association=an.get("association"),
            rle=an.get("segmentation"),
            bbox=tuple(an["bbox"]),
        )
        anns_by_image.setdefault(ann.image_id, []).append(ann)

    # group by association id per image
    groups: Dict[int, Dict[int, List[Ann]]] = {}
    for img_id, anns in anns_by_image.items():
        g: Dict[int, List[Ann]] = {}
        for a in anns:
            if a.association is None:
                # skip items without association
                continue
            g.setdefault(a.association, []).append(a)
        groups[img_id] = g
    return images, anns_by_image, groups


def resolve_pair_masks(
    root: Path, img: Img, anns: List[Ann]
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.height, img.width

    obj_mask = np.zeros((h, w), dtype=np.uint8)
    sh_mask = np.zeros((h, w), dtype=np.uint8)

    # Prefer per-annotation RLE; fallback to per-image mask PNGs if provided
    has_rle = any(a.rle for a in anns)

    if has_rle:
        for a in anns:
            m = to_bool_mask_from_rle(a.rle, h, w) if a.rle else np.zeros((h, w), dtype=np.uint8)
            if a.category_id == 1:
                obj_mask |= m
            elif a.category_id == 2:
                sh_mask |= m
    else:
        if img.object_mask_path:
            obj_mask |= load_mask_from_png(root / img.object_mask_path, (h, w))
        if img.shadow_mask_path:
            sh_mask |= load_mask_from_png(root / img.shadow_mask_path, (h, w))

    return obj_mask, sh_mask


def save_image_with_point(src: Path, dst: Path, point: Tuple[int, int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    r = 5  # radius of the dot
    x, y = point
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
    img.save(dst, quality=95)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--soba_root", type=Path, required=True, help="Path to SOBA root (folder containing SOBA/ and annotations/)")
    ap.add_argument("--ann_json", type=Path, required=True, help="Path to annotations JSON")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output dataset directory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    meta = load_json(args.ann_json)
    images, _, groups = build_indices(meta)

    out_images = args.out_dir / "images"
    out_masks = args.out_dir / "masks"
    out_shadow_masks = args.out_dir / "shadow_masks"
    out_object_masks = args.out_dir / "object_masks"
    out_points = args.out_dir / "points"

    sample_idx = 0
    for img_id, assoc_groups in groups.items():
        img = images[img_id]
        img_src = args.soba_root / img.file_name
        for assoc_id, anns in assoc_groups.items():
            obj_mask, sh_mask = resolve_pair_masks(args.soba_root, img, anns)

            if obj_mask.max() == 0 or sh_mask.max() == 0:
                # skip incomplete pairs
                continue

            point = sample_point(obj_mask, rng)
            union_mask = (obj_mask | sh_mask).astype(np.uint8)

            sample_name = f"{img_id:06d}_{assoc_id:03d}"
            img_dst_rel = f"images/{sample_name}.jpg"
            img_dst_rel_with_point = f"images_with_points/{sample_name}.jpg"

            save_image(img_src, args.out_dir / img_dst_rel)
            save_image_with_point(img_src, args.out_dir / img_dst_rel_with_point, point)
            save_mask_png(union_mask, out_masks / f"{sample_name}.png")
            save_mask_png(obj_mask, out_object_masks / f"{sample_name}.png")
            save_mask_png(sh_mask, out_shadow_masks / f"{sample_name}.png")
            save_point_json(img_dst_rel, point, out_points / f"{sample_name}.json")

            sample_idx += 1

    print(f"Saved {sample_idx} samples to {args.out_dir}")


if __name__ == "__main__":
    main()
