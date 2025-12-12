import argparse
import random
import shutil
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def load_mask_from_png(p: Path, shape: tuple[int, int] = None) -> np.ndarray:
    if not p.exists():
        p_jpg = p.with_suffix(".jpg")
        if p_jpg.exists():
            p = p_jpg
        else:
            if shape:
                return np.zeros(shape, dtype=np.uint8)
            raise FileNotFoundError(f"Mask not found: {p}")

    arr = np.array(Image.open(p).convert("L"))
    if shape is not None and arr.shape != shape:
        arr = np.array(Image.fromarray(arr).resize((shape[1], shape[0]), Image.NEAREST))
    return (arr > 0).astype(np.uint8)

def sample_point(mask: np.ndarray, rng: random.Random) -> tuple[int, int]:
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
    shutil.copy2(src, dst)

def save_point_json(img_rel: str, point: tuple[int, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"image": img_rel, "point": [point[0], point[1]]}, f, ensure_ascii=False)

def find_image_file(root_dir: Path, filename: str) -> Path:
    base_path = root_dir / filename
    if base_path.exists():
        return base_path
    jpg_path = base_path.with_suffix(".jpg")
    if jpg_path.exists():
        return jpg_path
    return None

def process_list(
    file_list_path: Path, 
    deroba_root: Path, 
    out_dir: Path, 
    rng: random.Random, 
    subset_name: str,
    limit: int | None = None
) -> int:
    print(f"\nProcessing list: {file_list_path}")
    
    with open(file_list_path) as f:
        filenames = [line.strip() for line in f.readlines() if line.strip()]

    if limit is not None and limit > 0:
        print(f"[INFO] Limiting {subset_name} set to {limit} samples (original: {len(filenames)})")
        filenames = filenames[:limit]

    src_images_dir = deroba_root / "original_image"
    src_fg_dir = deroba_root / "foreground_mask"
    src_refl_dir = deroba_root / "reflection_mask"

    dst_images = out_dir / "images"
    dst_masks = out_dir / "masks"
    dst_obj_masks = out_dir / "object_masks"
    dst_points = out_dir / "points"

    count = 0
    for fname in filenames:
        img_path = find_image_file(src_images_dir, fname)
        fg_path = src_fg_dir / fname
        refl_path = src_refl_dir / fname

        if not img_path:
            continue

        try:
            with Image.open(img_path) as tmp_img:
                w, h = tmp_img.size
                
            fg_mask = load_mask_from_png(fg_path, shape=(h, w))
            refl_mask = load_mask_from_png(refl_path, shape=(h, w))
        except Exception as e:
            print(f"  [ERR] Error processing {fname}: {e}")
            continue

        if fg_mask.max() == 0:
            continue

        try:
            point = sample_point(fg_mask, rng)
        except ValueError:
            continue

        union_mask = (fg_mask | refl_mask).astype(np.uint8)

        stem = Path(fname).stem
        final_fname = img_path.name
        
        save_image(img_path, dst_images / final_fname)
        save_mask_png(union_mask, dst_masks / f"{stem}.png")
        save_mask_png(fg_mask, dst_obj_masks / f"{stem}.png")
        
        img_rel_path = f"images/{final_fname}"
        save_point_json(img_rel_path, point, dst_points / f"{stem}.json")
        
        count += 1
        if count % 100 == 0:
            print(f"  Processed {count} images...")

    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deroba_root", type=Path, required=True)
    ap.add_argument("--train_txt", type=Path, required=True)
    ap.add_argument("--test_txt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    
    ap.add_argument("--limit", type=int, default=None, help="Limit number of samples for debugging")
    
    args = ap.parse_args()

    rng = random.Random(args.seed)

    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "object_masks").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "points").mkdir(parents=True, exist_ok=True)

    print(f"Starting processing. Root: {args.deroba_root}")
    if args.limit:
        print(f"!!! DEBUG MODE: Processing only {args.limit} images per list !!!")
    
    n_train = process_list(args.train_txt, args.deroba_root, args.out_dir, rng, "train", args.limit)
    print(f"--> Saved {n_train} training samples.")
    
    n_test = process_list(args.test_txt, args.deroba_root, args.out_dir, rng, "test", args.limit)
    print(f"--> Saved {n_test} test samples.")
    
    print(f"Total saved: {n_train + n_test} to {args.out_dir}")

if __name__ == "__main__":
    main()