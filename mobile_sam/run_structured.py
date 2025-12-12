import hydra
import torch
import torch.nn as nn
import torch_pruning as tp
from omegaconf import DictConfig, OmegaConf
import wandb
import os

from mobile_sam.common import setup_logger, get_logger, set_seed
from mobile_sam.data import build_dataloaders
from mobile_sam.model import load_mobilesam_vit_t
from mobile_sam.engine import Trainer

log = get_logger(__name__)

def get_ignored_layers(model: nn.Module) -> list[nn.Module]:
    """
    Identyfikuje warstwy, których NIE WOLNO prunować.
    """
    ignored_layers = []
    
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None and not m.weight.requires_grad:
                ignored_layers.append(m)
    
    if hasattr(model.image_encoder, 'neck'):
        print("Locking MobileSAM Neck layers to preserve output dimension...")
        for m in model.image_encoder.neck.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
                ignored_layers.append(m)
                
    return ignored_layers

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the structured pruning pipeline using torch-pruning.
    
    Performs iterative channel/layer pruning based on dependency graphs, 
    followed by fine-tuning and performance benchmarking.
    """
    setup_logger()
    set_seed(cfg.system.seed)
    
    if not cfg.pruning.enabled:
        log.error("Pruning disabled in config.")
        return

    log.info("--- PHASE 2a: Structured Pruning (Torch-Pruning) ---")

    train_loader, val_loader = build_dataloaders(
        cfg_data=cfg.data,
        val_split=cfg.train.val_split,
        num_workers=cfg.system.num_workers,
        seed=cfg.system.seed
    )
    
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
            tags=["pruning", "structured"]
        )

    target_sparsity = cfg.pruning.target_sparsity
    iterations = cfg.pruning.iterations
    
    ignored_layers = get_ignored_layers(model)
    example_input = torch.randn(1, 3, 1024, 1024).to(cfg.system.device)
    
    for name, m in model.image_encoder.named_modules():
        if "attn" in name and isinstance(m, (nn.Linear, nn.Conv2d)):
            ignored_layers.append(m)
    
    imp = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model.image_encoder,
        example_inputs=example_input,
        importance=imp,
        iterative_steps=iterations,
        ch_sparsity=target_sparsity,
        ignored_layers=ignored_layers,
        round_to=64
    )

    original_params = sum(p.numel() for p in model.image_encoder.parameters())
    total_epochs = 0
    
    for i in range(1, iterations + 1):
        log.info(f"\n[Structured Pruning Iteration {i}/{iterations}]")
        
        pruner.step()
        
        current_params = sum(p.numel() for p in model.image_encoder.parameters())
        real_sparsity = 1.0 - (current_params / original_params)
        log.info(f"Current Encoder Params: {current_params/1e6:.2f}M")

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
            patience=cfg.pruning.patience,
            custom_sparsity=real_sparsity,
            artefact_weight=cfg.train.artefact_loss_weight
        )
        
        ckpt_name = f"checkpoints/struct_iter_{i}.pt"
        trainer.fit(ckpt_name)
        total_epochs += ft_epochs

    final_path = f"models/final_structured_{int(target_sparsity*100)}sp.pt"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(model.state_dict(), final_path)
    log.info(f"Saved final structured model: {final_path}")

    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    main()