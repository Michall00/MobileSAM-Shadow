import albumentations as A  # noqa: N812
import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, random_split

from mobile_sam.utils.dataset_utils import (
    TASK_NORMAL,
    TASK_REFLECTION,
    TASK_SHADOW,
    two_mask_collate,
)
from mobile_sam.utils.obj_prompt_reflection_dataset import ObjPromptReflectionDataset
from mobile_sam.utils.obj_prompt_shadow_dataset import ObjPromptShadowDataset
from mobile_sam.common import get_logger


log = get_logger(__name__)


def build_dataloaders(
    cfg_data: DictConfig,
    val_split: float,
    num_workers: int,
    seed: int,
    augmenter: A.Compose | None = None
) -> tuple[DataLoader, DataLoader | None]:
    
    datasets = []
    
    if cfg_data.get("shadow", None):
        log.info("Initializing Shadow Dataset...")
        ds_shadow = ObjPromptShadowDataset(
            images_dir=cfg_data.shadow.images_dir,
            object_masks_dir=cfg_data.shadow.obj_masks_dir,
            target_masks_dir=cfg_data.shadow.masks_dir,
            size=cfg_data.size,
            seed=seed,
            return_obj_mask=True,
            augmenter=augmenter
            # task_id=TASK_SHADOW
        )
        datasets.append(ds_shadow)

    if cfg_data.get("reflection", None):
        log.info("Initializing Reflection Dataset...")
        ds_refl = ObjPromptReflectionDataset(
            images_dir=cfg_data.reflection.images_dir,
            object_masks_dir=cfg_data.reflection.obj_masks_dir,
            target_masks_dir=cfg_data.reflection.masks_dir,
            size=cfg_data.size,
            seed=seed,
            return_obj_mask=True,
            augmenter=augmenter,
            task_id=TASK_REFLECTION
        )
        datasets.append(ds_refl)
        
    if cfg_data.get("normal", None):
        log.info("Initializing Normal Dataset...")
        ds_normal = ObjPromptReflectionDataset(
            images_dir=cfg_data.normal.images_dir,
            object_masks_dir=cfg_data.normal.obj_masks_dir,
            target_masks_dir=cfg_data.normal.masks_dir,
            size=cfg_data.size,
            seed=seed,
            return_obj_mask=False,
            augmenter=augmenter,
            task_id=TASK_NORMAL
        )
        datasets.append(ds_normal)

    if not datasets:
        raise ValueError("No datasets configured! Check config.yaml")

    if len(datasets) > 1:
        log.info(f"Combining {len(datasets)} datasets.")
        full_ds = ConcatDataset(datasets)
    else:
        full_ds = datasets[0]

    if val_split <= 0.0:
        return DataLoader(full_ds, batch_size=cfg_data.batch_size, shuffle=True, num_workers=num_workers, collate_fn=two_mask_collate), None
    
    val_len = int(len(full_ds) * val_split)
    train_len = len(full_ds) - val_len
    
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=g)
    
    log.info(f"Combined Data: Train={len(train_ds)}, Val={len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=cfg_data.batch_size, shuffle=True, num_workers=num_workers, collate_fn=two_mask_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg_data.batch_size, shuffle=False, num_workers=num_workers, collate_fn=two_mask_collate)
    
    return train_loader, val_loader