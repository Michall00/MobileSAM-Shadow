
import argparse
import json
import random
import logging
from pathlib import Path
from PIL import Image
import numpy as np
from lvis import LVIS
import requests

def extract_instance_lvis(lvis, img_root, img_info, ann):
    file_name = img_info["coco_url"].split("/")[-1]
    img_path = img_root / file_name
    if not img_path.exists():
        logging.warning(f"Image not found locally: {img_path}, downloading from {img_info['coco_url']}")
        try:
            response = requests.get(img_info["coco_url"], timeout=10)
            response.raise_for_status()
            img_path.parent.mkdir(parents=True, exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            logging.error(f"Failed to download image from {img_info['coco_url']}: {e}")
            raise FileNotFoundError(f"Image not found and download failed: {img_path}")
    img = Image.open(img_path).convert("RGB")
    m = lvis.ann_to_mask(ann)
    mask = Image.fromarray((m.astype(np.uint8) * 255), mode="L")
    x, y, w, h = [int(v) for v in ann["bbox"]]
    crop_img = img.crop((x, y, x + w, y + h))
    crop_mask = mask.crop((x, y, x + w, y + h))
    return crop_img, crop_mask, int(ann["category_id"])

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser("LVIS object cutout and paste")
    parser.add_argument("--lvis-ann", type=Path, required=True)
    parser.add_argument("--img-root", type=Path, required=True)
    parser.add_argument("--bg-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--allowed-category-ids", type=int, nargs="*", default=None, help="List of allowed category_ids")
    parser.add_argument("--min-mask-area", type=int, default=0, help="Minimal mask area in pixels")
    args = parser.parse_args()

    logging.info(f"Loading LVIS annotations from {args.lvis_ann}")
    lvis = LVIS(str(args.lvis_ann))
    anns_by_img = {}
    for ann in lvis.anns.values():
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        img_id = int(ann["image_id"])
        anns_by_img.setdefault(img_id, []).append(ann)
    logging.info(f"Found {len(anns_by_img)} images with annotations.")

    bg_paths = [p for p in args.bg_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    logging.info(f"Found {len(bg_paths)} background images.")
    (args.out_root / "images").mkdir(parents=True, exist_ok=True)
    (args.out_root / "masks").mkdir(parents=True, exist_ok=True)
    (args.out_root / "cutouts").mkdir(parents=True, exist_ok=True)
    meta_path = args.out_root / "annotations.jsonl"

    saved = 0
    trials = 0
    max_trials = args.n * 20
    with open(meta_path, "w", encoding="utf-8") as fh:
        while saved < args.n and trials < max_trials:
            trials += 1
            if not anns_by_img:
                logging.error("No images with annotations available.")
                break
            img_id = random.choice(list(anns_by_img.keys()))
            img_info = lvis.imgs[img_id]
            anns = anns_by_img[img_id]
            ann = random.choice(anns)
            try:
                obj_img, obj_mask, cat_id = extract_instance_lvis(lvis, args.img_root, img_info, ann)
            except Exception as e:
                logging.warning(f"Failed to extract instance: {e}. img_info: {img_info}")
                continue
            # Filtrowanie po kategorii
            if args.allowed_category_ids is not None and cat_id not in args.allowed_category_ids:
                continue
            # Filtrowanie po powierzchni maski
            mask_np = np.array(obj_mask)
            mask_area = np.sum(mask_np > 0)
            if mask_area < args.min_mask_area:
                continue
            if not bg_paths:
                logging.error("No background images found.")
                break
            bg_path = random.choice(bg_paths)
            try:
                bg = Image.open(bg_path).convert("RGB")
            except Exception as e:
                logging.warning(f"Failed to open background image {bg_path}: {e}")
                continue
            # Resize wszystko do 1024x1024
            target_size = (1024, 1024)
            obj_img = obj_img.resize(target_size, Image.BICUBIC)
            obj_mask = obj_mask.resize(target_size, Image.NEAREST)
            bg = bg.resize(target_size, Image.BICUBIC)
            pasted = bg.copy()
            pasted.paste(obj_img, (0, 0), obj_mask)
            img_name = f"sample_{saved:05d}.jpg"
            mask_name = f"sample_{saved:05d}_mask.png"
            cutout_name = f"sample_{saved:05d}_cutout.png"
            try:
                pasted.save(args.out_root / "images" / img_name)
                obj_mask.save(args.out_root / "masks" / mask_name)
                cutout = Image.new("RGBA", target_size, (0,0,0,0))
                cutout.paste(obj_img, (0,0), obj_mask)
                cutout.save(args.out_root / "cutouts" / cutout_name)
            except Exception as e:
                logging.warning(f"Failed to save images for sample {saved}: {e}")
                continue
            meta = {
                "image": str(args.out_root / "images" / img_name),
                "mask": str(args.out_root / "masks" / mask_name),
                "cutout": str(args.out_root / "cutouts" / cutout_name),
                "category_id": cat_id,
                "background": str(bg_path),
                "source_image": str(args.img_root / img_info["coco_url"].split("/")[-1]),
                "mask_area": int(mask_area),
            }
            fh.write(json.dumps(meta) + "\n")
            saved += 1
            if saved % 10 == 0:
                logging.info(f"Saved {saved}/{args.n} samples.")
    logging.info(f"Done. Saved {saved} samples.")

if __name__ == "__main__":
    main()
