import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
import wandb

from mobile_sam.prune import apply_pruning, remove_pruning_reparam
from mobile_sam.utils.dataset_utils import AugmentationConfig
from mobile_sam.common import get_logger, set_seed, setup_logger
from mobile_sam.model import load_mobilesam_vit_t
from mobile_sam.data import build_dataloaders
from mobile_sam.engine import Trainer

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
        augmenter=augmenter
    )
    log.info(f"Train dataset size: {len(train_loader.dataset)} samples")

    wandb_run = None
    if cfg.wandb.enabled:
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            config=config_dict
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
    )

    if cfg.train.resume_ckpt:
        log.info(f"Loading checkpoint from {cfg.train.resume_ckpt}")
        checkpoint = torch.load(cfg.train.resume_ckpt, map_location=cfg.system.device)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            elif all(isinstance(k, str) and k.startswith(("image_encoder.", "prompt_encoder.", "mask_decoder.")) for k in checkpoint.keys()):
                model.load_state_dict(checkpoint)
            elif all(k in checkpoint for k in ("image_encoder", "prompt_encoder", "mask_decoder")):
                flat_state = {}
                for submodule in ("image_encoder", "prompt_encoder", "mask_decoder"):
                    subdict = checkpoint[submodule]
                    for k, v in subdict.items():
                        flat_state[f"{submodule}.{k}"] = v
                model.load_state_dict(flat_state, strict=False)
                log.info("Loaded checkpoint (image_encoder, prompt_encoder, mask_decoder)")
            else:
                log.error(f"Checkpoint does not contain 'model' key or does not look like a model state_dict. Keys: {list(checkpoint.keys())}")
                raise RuntimeError("Invalid checkpoint format!")
        else:
            log.error("Checkpoint is not a dictionary!")
            raise RuntimeError("Invalid checkpoint format!")
        try:
            if isinstance(checkpoint, dict):
                if "optimizer" in checkpoint:
                    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                if "scheduler" in checkpoint:
                    trainer.scheduler.load_state_dict(checkpoint["scheduler"])
            log.info("Optimizer and scheduler state loaded.")
        except Exception as e:
            log.warning(f"Could not load optimizer/scheduler state: {e}")
    elif cfg.model.pretrained_path:
        log.info(f"Loading pretrained weights from {cfg.model.pretrained_path}")
        state = torch.load(cfg.model.pretrained_path, map_location=cfg.system.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    
    log.info("--- PHASE 1: Base Training ---")

    trainer.fit(cfg.train.output_ckpt)
    
    total_epochs_passed += cfg.train.epochs

    torch.save(trainer.model.state_dict(), "model_base_full_backup.pt")

    PRUNE_ENABLE = True
    ITERATIONS = 15
    AMOUNT_PER_ITER = 0.05
    
    REWIND_LR = 1e-4
    L1_LAMBDA = 1e-5

    if PRUNE_ENABLE:
        log.info(f"START ITERATIVE PRUNING: {ITERATIONS} iterations of {AMOUNT_PER_ITER*100}%")

        for i in range(1, ITERATIONS + 1):
            log.info(f"\n[PRUNING] Iteration {i}/{ITERATIONS}...")
            
            target_sparsity = 1.0 - ((1.0 - AMOUNT_PER_ITER) ** i)
            
            if target_sparsity < 0.2:
                ft_epochs = 2
            elif target_sparsity < 0.5:
                ft_epochs = 3
            else:
                ft_epochs = 5
            
            apply_pruning(
                model=trainer.model,
                mode="layer_l1_unstructured", 
                amount=AMOUNT_PER_ITER,
                include_linear=True,
                structured_n=1,
                structured_dim=0,
            )

            trainer_ft = Trainer(
                model=trainer.model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=REWIND_LR,
                weight_decay=cfg.train.weight_decay,
                max_epochs=ft_epochs,
                grad_clip=cfg.train.grad_clip,
                amp=cfg.train.amp,
                device=cfg.system.device,
                epoch_offset=total_epochs_passed,
                vis_dir=cfg.test.vis_dir,
                vis_every=cfg.test.vis_every,
                vis_num=cfg.test.vis_num,
                wandb_run=wandb_run,
                wandb_images=cfg.wandb.wandb_images_num,
                l1_lambda=L1_LAMBDA,
            )

            ft_ckpt_name = f"model_pruned_iter_{i}.pt"
            trainer_ft.fit(ft_ckpt_name)
            
            total_epochs_passed += ft_epochs

        log.info("Removing pruning reparameterization (burning zeros)...")
        remove_pruning_reparam(trainer.model)
        
        final_name = f"models/model_final_pruned_{ITERATIONS}x{int(AMOUNT_PER_ITER*100)}.pt"
        torch.save(trainer.model.state_dict(), final_name)
        log.info(f"Saved final model: {final_name}")

    if wandb_run:
        wandb_run.finish()
        
if __name__ == "__main__":
    main()
