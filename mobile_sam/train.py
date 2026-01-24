import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import wandb
from mobile_sam.common import get_logger, set_seed, setup_logger
from mobile_sam.data import build_dataloaders
from mobile_sam.engine import Trainer
from mobile_sam.model import load_mobilesam_vit_t
from mobile_sam.utils.dataset_utils import AugmentationConfig

setup_logger()
log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    set_seed(cfg.system.seed)

    model = load_mobilesam_vit_t(None, device=cfg.system.device)

    augmenter = None
    if cfg.data.augment:
        aug_params = OmegaConf.to_container(cfg.data.aug_params, resolve=True)
        augmenter = AugmentationConfig(aug_params)

    train_loader, val_loader = build_dataloaders(
        cfg_data=cfg.data,
        val_split=cfg.train.val_split,
        num_workers=cfg.system.num_workers,
        seed=cfg.system.seed,
        augmenter=augmenter,
    )
    log.info(f"Train dataset size: {len(train_loader.dataset)} samples")

    run_name = f"BASE_Ep{cfg.train.epochs}_LR{cfg.train.lr:.0e}_BS{cfg.data.batch_size}"
    wandb_run = None
    if cfg.wandb.enabled:
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            name=run_name,
            config=config_dict,
        )

    total_epochs_passed = 0

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        max_epochs=cfg.train.epochs,
        grad_clip=1.0,
        amp=cfg.train.amp,
        device=cfg.system.device,
        epoch_offset=total_epochs_passed,
        vis_dir=cfg.test.vis_dir,
        vis_every=cfg.test.vis_every,
        vis_num=cfg.test.vis_num,
        wandb_run=wandb_run,
        wandb_images=cfg.wandb.wandb_images_num,
        tuning_strategy=cfg.train.tuning_strategy,
        qat_enabled=cfg.train.qat_enabled,
        qat_type=cfg.train.qat_type,
    )

    log.info(f"Loading pretrained weights from {cfg.model.pretrained_path}")
    state = torch.load(cfg.model.pretrained_path, map_location=cfg.system.device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)

    log.info("--- PHASE 1: Base Training ---")

    trainer.fit(cfg.train.output_ckpt)

    total_epochs_passed += cfg.train.epochs

    torch.save(trainer.model.state_dict(), "model_base.pt")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
