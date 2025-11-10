from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from mobile_sam.utils.obj_prompt_shadow_dataset import ObjPromptShadowDataset, AugmentationConfig

images_dir = Path("/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/images")
obj_masks_dir = Path("/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/object_masks")
tgt_masks_dir = Path("/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/masks")

out_images_dir = Path("/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/images_aug")
out_obj_dir = Path("/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/object_masks_aug")
out_tgt_dir = Path("/home/msadowski/Studia/Inzynierka/MobileSAM/dataset/prepared_soba/masks_aug")

out_images_dir.mkdir(parents=True, exist_ok=True)
out_obj_dir.mkdir(parents=True, exist_ok=True)
out_tgt_dir.mkdir(parents=True, exist_ok=True)

augment_cfgs = [
    ("flip", AugmentationConfig(use_flip=True,  use_brightness=False, use_rotate=False, use_blur=False, use_hue=False)),
    ("rotate", AugmentationConfig(use_flip=False,  use_brightness=False, use_rotate=True,  use_blur=False, use_hue=False)),
    ("bright", AugmentationConfig(use_flip=False,  use_brightness=True,  use_rotate=False,  use_blur=False, use_hue=False)),
    ("blur", AugmentationConfig(use_flip=False,  use_brightness=False,  use_rotate=False,  use_blur=True,  use_hue=False)),
    ("hue", AugmentationConfig(use_flip=False,  use_brightness=False,  use_rotate=False,  use_blur=False,  use_hue=True)),
]

dataset = ObjPromptShadowDataset(
    images_dir=images_dir,
    obj_masks_dir=obj_masks_dir,
    target_masks_dir=tgt_masks_dir,
    size=1024,
    return_obj_mask=True,
    augmenter=None
)

print(f"Generating {len(augment_cfgs)} deterministic augmentations per sample...")

for i in tqdm(range(len(dataset))):
    ipath, opath, tpath, _ = dataset.samples[i]
    base = Path(ipath).stem

    img = Image.open(ipath).convert("RGB")
    obj = Image.open(opath).convert("L")
    tgt = Image.open(tpath).convert("L")

    obj_np = np.array(obj)
    tgt_np = np.array(tgt)

    for k, (cfg_name, augmenter) in enumerate(augment_cfgs):
        aug_img, aug_obj, aug_tgt = augmenter(np.array(img), np.array(obj_np.copy()), np.array(tgt_np.copy()))

        out_img_path = out_images_dir / cfg_name / f"{base}_{k}.jpg"
        out_obj_path = out_obj_dir / cfg_name / f"{base}_{k}.png"
        out_tgt_path = out_tgt_dir / cfg_name / f"{base}_{k}.png"

        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        out_obj_path.parent.mkdir(parents=True, exist_ok=True)
        out_tgt_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(aug_img).save(out_img_path)
        Image.fromarray(aug_obj).save(out_obj_path)
        Image.fromarray(aug_tgt).save(out_tgt_path)

print("Augmentation completed.")