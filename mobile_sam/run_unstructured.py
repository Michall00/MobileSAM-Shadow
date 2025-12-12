import hydra
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from omegaconf import DictConfig, OmegaConf
import wandb
import os

from mobile_sam.common import setup_logger, get_logger, set_seed
from mobile_sam.data import build_dataloaders
from mobile_sam.model import load_mobilesam_vit_t
from mobile_sam.engine import Trainer

log = get_logger(__name__)

def apply_unstructured_pruning(model: nn.Module, amount: float) -> None:
    """
    Applies global unstructured L1 pruning to all Conv2d and Linear layers 
    within the image encoder.
    """
    params_to_prune = []
    
    for name, module in model.image_encoder.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params_to_prune.append((module, 'weight'))
            
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

def make_permanent(model: nn.Module) -> None:
    """
    Removes the pruning reparameterization (masks), making the zeroed weights 
    permanent in the model state dictionary.
    """
    for name, module in model.image_encoder.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if prune.is_pruned(module):
                prune.remove(module, 'weight')

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the unstructured pruning pipeline.
    
    Performs iterative global L1 pruning followed by fine-tuning and benchmarking.
    """
    setup_logger()
    set_seed(cfg.system.seed)
    
    if not cfg.pruning.enabled:
        log.error("Pruning disabled in config.")
        return

    log.info("--- PHASE 2b: Unstructured Pruning (L1) ---")

    train_loader, val_loader = build_dataloaders(
        cfg_data=cfg.data,
        val_split=cfg.train.val_split,
        num_workers=cfg.system.num_workers,
        seed=cfg.system.seed
    )

    log.info(f"Loading base model: {cfg.train.resume_ckpt}")
    model = load_mobilesam_vit_t(None, device=cfg.system.device)

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
    elif cfg.model.pretrained_path:
        log.info(f"Loading pretrained weights from {cfg.model.pretrained_path}")
        state = torch.load(cfg.model.pretrained_path, map_location=cfg.system.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)


    run_name = f"BASE_Ep{cfg.train.epochs}_LR{cfg.train.lr:.0e}_BS{cfg.data.batch_size}"
    wandb_run = None
    if cfg.wandb.enabled:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=["pruning", "unstructured"]
        )

    iterations = cfg.pruning.iterations
    amount_per_iter = cfg.pruning.amount_per_iter
    total_epochs = 0
    
    for i in range(1, iterations + 1):
        log.info(f"\n[Unstructured Pruning Iteration {i}/{iterations}]")
        
        apply_unstructured_pruning(model, amount=amount_per_iter)
        
        total_params = 0
        zero_params = 0
        
        for module in model.modules():
            if hasattr(module, "weight_mask"):
                mask = module.weight_mask
                total_params += mask.numel()
                zero_params += torch.sum(mask == 0).item()
                
            elif hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
                total_params += module.weight.numel()
        current_sparsity = 0.0 if total_params == 0 else zero_params / total_params
        log.info(f"Current Sparsity: {current_sparsity:.2%}")

        ft_epochs = cfg.pruning.finetune_epochs
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=cfg.pruning.lr,
            max_epochs=ft_epochs,
            device=cfg.system.device,
            epoch_offset=total_epochs,
            wandb_run=wandb_run,
            patience=cfg.pruning.get("patience", 3)
        )
        
        ckpt_name = f"checkpoints/unstruct_iter_{i}.pt"
        trainer.fit(ckpt_name)
        
        speed_stats = trainer.benchmark(num_runs=50)
        if wandb_run:
            wandb_run.log({
                "pruning/fps": speed_stats["fps"],
                "sparsity": current_sparsity,
                "global_epoch": total_epochs
            })
            
        total_epochs += ft_epochs

    make_permanent(model)
    final_path = f"models/final_unstructured_{int(current_sparsity*100)}sp.pt"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(model.state_dict(), final_path)
    log.info(f"Saved final unstructured model: {final_path}")

    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    main()