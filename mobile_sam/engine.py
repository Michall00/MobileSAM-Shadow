import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Literal
import segmentation_models_pytorch as smp
from mobile_sam.model import freeze_non_encoder
from mobile_sam.utils.eval import compute_metrics, forward_mobile_sam
import os
from mobile_sam.utils.dataset_utils import (
    TASK_REFLECTION,
    TASK_SHADOW,
)
from mobile_sam.utils.common import sam_denormalize_float, make_panel_with_points
from mobile_sam.common import get_logger 
from PIL import ImageDraw
import wandb
from tqdm import tqdm
import math
import time

from torchao.quantization.qat import (
    QATConfig
)
from torchao.quantization import Int8DynamicActivationInt4WeightConfig
from torchao.quantization.quant_api import quantize_


log = get_logger(__name__)

class ArtefactLoss(nn.Module):
    def __init__(self, pos_weight: float = 3.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, target) + self.dice(logits, target)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        lr: float = 1e-4,
        weight_decay: float = 5e-2,
        max_epochs: int = 10,
        grad_clip: float = 1.0,
        amp: bool = True,
        device: str = "cuda",
        epoch_offset: int = 0,
        vis_dir: str | None = None,
        vis_every: int = 5,
        vis_num: int = 8,
        wandb_run: Any | None = None,
        wandb_images: int = 16,
        perform_baseline_eval: bool = True,
        l1_lambda: float = 0.0,
        patience: int = 10,
        custom_sparsity: float | None = None,
        artefact_weight: float = 1.0,
        scheduler_type: str = "cosine",
        tuning_strategy: Literal["encoder_only", "full_model"] = "encoder_only",
        qat_enabled: bool = False,
        qat_type: str = "int8_dyn_act_int4_weight",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = ArtefactLoss(pos_weight=artefact_weight).to(device)
        self.device = device
        
        self.qat_enabled = qat_enabled

        if self.qat_enabled:
            log.info("Preparing model for Int8 Weight + Int8 Activation QAT")
            
            self.base_config = Int8DynamicActivationInt4WeightConfig(group_size=32)
            qat_config = QATConfig(self.base_config, step="prepare")
            self.model.to("cpu")
            quantize_(self.model, qat_config)
            self.model.to(self.device)

        if tuning_strategy == "encoder_only":
            params_to_optimize = freeze_non_encoder(self.model)
            log.info("Strategy: Training ENCODER ONLY (Decoder frozen)")
            
        elif tuning_strategy == "full_model":
            for param in self.model.parameters():
                param.requires_grad = True
            params_to_optimize = self.model.parameters()
            log.info("Strategy: Training FULL MODEL (Encoder + Decoder)")
            
        else:
            raise ValueError(f"Unknown tuning strategy: {tuning_strategy}")

        self.optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))


        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        elif scheduler_type == "constant":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=9999, gamma=1.0)
        else:
            log.warning(f"Unknown scheduler type '{scheduler_type}', defaulting to cosine.")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)

        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.amp = amp
        self.epoch_offset = epoch_offset
        self.patience = patience
        
        enable_amp = bool(amp and self.device == "cuda")
        try:
            self.scaler = torch.amp.GradScaler(self.device, enabled=enable_amp)
        except TypeError:
            self.scaler = torch.amp.GradScaler(enabled=enable_amp)
            
        self.vis_dir = vis_dir
        self.vis_every = vis_every
        self.vis_num = vis_num
        self.wandb = wandb_run
        self.wandb_images = wandb_images
        self.perform_baseline_eval = perform_baseline_eval
        self.l1_lambda = l1_lambda
        self.custom_sparsity = custom_sparsity
        
        self.define_params_to_regularize()
        
    def define_params_to_regularize(self) -> None:
        self.params_to_regularize = [
            p for name, p in self.model.named_parameters()
            if p.requires_grad 
            and 'bias' not in name 
            and 'norm' not in name 
            and 'bn' not in name
        ]
    
    def save_predictions(self, epoch: int) -> None:
        if not self.vis_dir:
            return
        out_dir = os.path.join(self.vis_dir, f"epoch_{epoch:03d}")
        os.makedirs(out_dir, exist_ok=True)

        loader = self.val_loader if self.val_loader is not None else self.train_loader
        self.model.eval()

        saved = 0
        wb_imgs: list[Any] = []

        with torch.no_grad():
            for b_idx, batch in enumerate(loader):
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                points = [p.to(self.device) for p in batch["points"]]
                labels = [l.to(self.device) for l in batch["point_labels"]]
                boxes = [b.to(self.device) for b in batch["boxes"]]

                logits = forward_mobile_sam(self.model, images, points, labels, boxes)
                preds = torch.sigmoid(logits) >= 0.5

                for i in range(images.size(0)):
                    img_np = sam_denormalize_float(images[i])
                    gt_np = masks[i, 0].cpu().numpy() > 0.5
                    pred_np = preds[i, 0].cpu().numpy() > 0.5

                    box = boxes[i]
                    if box.numel() == 4:
                        x0, y0, x1, y1 = box.cpu().numpy()
                        panel = make_panel_with_points(img_np, gt_np, pred_np, points[i].cpu().numpy())
                        draw = ImageDraw.Draw(panel)
                        draw.rectangle([x0, y0, x1, y1], outline="blue", width=2)
                    else:
                        panel = make_panel_with_points(img_np, gt_np, pred_np, points[i].cpu().numpy())

                    try:
                        if self.wandb and saved < self.wandb_images:
                            wb_imgs.append(wandb.Image(panel, caption=f"Sample {saved}"))
                    except Exception as e:
                        log.warning(f"Failed to log image to WandB: {e}")

                    saved += 1
                    if saved >= self.vis_num:
                        break
                if saved >= self.vis_num:
                    break

        if self.wandb and wb_imgs:
            self.wandb.log({"predictions": wb_imgs}, step=epoch)

        log.info(f"Vis: Saved {saved} images to {out_dir}")
        self.model.train()
        
    def log_qat_stats(self, epoch: int):
        stats = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_fake_quant'):
                stats[f"qat/scale_{name}"] = module.weight_fake_quant.scale.mean().item()
                
        if self.wandb:
            self.wandb.log(stats, step=epoch)
            
    def baseline_evaluation(self) -> None:
        log.info(f"[Baseline Eval] Starting evaluation (Global Epoch: {self.epoch_offset})...")
        val_stats = self.validate(epoch=self.epoch_offset)
        sparsity = self.custom_sparsity if self.custom_sparsity is not None else self.calculate_sparsity()

        if self.wandb:
            self.wandb.log({
                "epoch": self.epoch_offset,
                "val/loss": val_stats["val_loss"],
                "val/f1": val_stats["val_f1"],
                "val/iou": val_stats["val_iou"],
                "val/ber": val_stats["val_ber"],
                "val/shadow_iou": val_stats.get("val_shadow_iou", float("nan")),
                "sparsity": sparsity,
                "model/lr": self.optimizer.param_groups[0]["lr"],
                "is_baseline": 1
            })
            
        log.info(f"[Baseline Eval] F1: {val_stats['val_f1']:.4f} | Sparsity: {sparsity:.2%} | Loss: {val_stats['val_loss']:.4f} | Global Epoch: {self.epoch_offset}")

    def _update_metrics(self, aggregator: dict, metrics: dict, prefix: str = ""):
        for k, v in metrics.items():
            key = f"{prefix}{k}" if prefix else k
            if key not in aggregator:
                aggregator[key] = 0.0
            aggregator[key] += v

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        agg = {
            "iou": 0.0, "f1": 0.0, "ber": 0.0, 
            "shadow_iou": 0.0, "shadow_f1": 0.0,
            "reflection_iou": 0.0, "reflection_f1": 0.0
        }
        counts = {"total": 0, "shadow": 0, "reflection": 0}
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            obj_mask = batch["obj_mask"].to(self.device, non_blocking=True) if "obj_mask" in batch else None
            task_ids = batch["task_id"].to(self.device)
            
            points = [p.to(self.device) for p in batch["points"]]
            labels = [l.to(self.device) for l in batch["point_labels"]]
            boxes = [b.to(self.device) for b in batch["boxes"]]

            with torch.amp.autocast(self.device, enabled=(self.amp and self.device == "cuda")):
                logits = forward_mobile_sam(self.model, images, points, labels, boxes)
                main_loss = self.loss_fn(logits, masks)
                
                l1_reg = torch.tensor(0.0, device=self.device)
                if self.l1_lambda > 0.0:
                    l1_reg = sum(p.abs().sum() for p in self.params_to_regularize)
                            
                loss = main_loss + (self.l1_lambda * l1_reg)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.image_encoder.parameters(), self.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(main_loss.item())
            l1_loss_accum = float(l1_reg.item())
            n_batches += 1
            counts["total"] += 1

            m_global = compute_metrics(logits, masks, thr=0.5)
            self._update_metrics(agg, m_global)

            shadow_mask = (task_ids == TASK_SHADOW)
            if shadow_mask.any():
                m_shadow = compute_metrics(
                    logits[shadow_mask], 
                    masks[shadow_mask], 
                    thr=0.5, 
                    obj_mask=obj_mask[shadow_mask] if obj_mask is not None else None
                )
                self._update_metrics(agg, m_shadow, prefix="shadow_")
                counts["shadow"] += 1

            refl_mask = (task_ids == TASK_REFLECTION)
            if refl_mask.any():
                m_refl = compute_metrics(
                    logits[refl_mask], 
                    masks[refl_mask], 
                    thr=0.5, 
                    obj_mask=obj_mask[refl_mask] if obj_mask is not None else None
                )
                self._update_metrics(agg, m_refl, prefix="reflection_")
                counts["reflection"] += 1

        prefix = "train"
        out = {f"{prefix}_loss": total_loss / max(1, n_batches)}
        out[f"{prefix}_l1_norm"] = l1_loss_accum / max(1, n_batches)
        
        for metric in ["iou", "f1", "ber"]:
            out[f"{prefix}_{metric}"] = agg[metric] / max(1, counts["total"])
            
        if counts["shadow"] > 0:
            out[f"{prefix}_shadow_iou"] = agg["shadow_iou"] / counts["shadow"]
            out[f"{prefix}_shadow_f1"] = agg["shadow_f1"] / counts["shadow"]
            
        if counts["reflection"] > 0:
            out[f"{prefix}_reflection_iou"] = agg["reflection_iou"] / counts["reflection"]
            out[f"{prefix}_reflection_f1"] = agg["reflection_f1"] / counts["reflection"]
            
        out["global_epoch"] = self.epoch_offset + epoch
        return out
    
    @torch.no_grad()
    def benchmark(self, num_runs: int = 100, warm_up: int = 10) -> dict[str, float]:
        self.model.eval()
        dummy_input = torch.randn(1, 3, 1024, 1024).to(self.device)
        
        for _ in range(warm_up):
            self.model.image_encoder(dummy_input)
            
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()

        for _ in range(num_runs):
            self.model.image_encoder(dummy_input)
            
        if self.device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        log.info(f"Benchmark Speed: {avg_time*1000:.2f} ms/img | {fps:.2f} FPS")
        return {"inference_ms": avg_time * 1000, "fps": fps}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            return {"val_loss": math.nan, "val_iou": math.nan, "val_f1": math.nan, "val_ber": math.nan}
        
        self.model.eval()
        t0 = time.perf_counter()
        
        total_loss = 0.0
        n_batches = 0
        agg = {"iou": 0.0, "f1": 0.0, "ber": 0.0, "shadow_iou": 0.0, "shadow_f1": 0.0, "reflection_iou": 0.0, "reflection_f1": 0.0}
        counts = {"total": 0, "shadow": 0, "reflection": 0}
        
        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            task_ids = batch["task_id"].to(self.device)
            points = [p.to(self.device) for p in batch["points"]]
            labels = [l.to(self.device) for l in batch["point_labels"]]
            boxes = [b.to(self.device) for b in batch["boxes"]]
            obj_mask = batch.get("obj_mask", None)
            if obj_mask is not None:
                obj_mask = obj_mask.to(self.device, non_blocking=True)
            
            logits = forward_mobile_sam(self.model, images, points, labels, boxes)
            loss = self.loss_fn(logits, masks)
            
            total_loss += float(loss.item())
            n_batches += 1
            counts["total"] += 1

            m_global = compute_metrics(logits, masks, thr=0.5)
            self._update_metrics(agg, m_global)
            
            shadow_mask = (task_ids == TASK_SHADOW)
            if shadow_mask.any():
                m_shadow = compute_metrics(logits[shadow_mask], masks[shadow_mask], thr=0.5, obj_mask=obj_mask[shadow_mask] if obj_mask is not None else None)
                self._update_metrics(agg, m_shadow, prefix="shadow_")
                counts["shadow"] += 1

            refl_mask = (task_ids == TASK_REFLECTION)
            if refl_mask.any():
                m_refl = compute_metrics(logits[refl_mask], masks[refl_mask], thr=0.5, obj_mask=obj_mask[refl_mask] if obj_mask is not None else None)
                self._update_metrics(agg, m_refl, prefix="reflection_")
                counts["reflection"] += 1

        prefix = "val"
        out = {f"{prefix}_loss": total_loss / max(1, n_batches)}
        for metric in ["iou", "f1", "ber"]:
            out[f"{prefix}_{metric}"] = agg[metric] / max(1, counts["total"])
        if counts["shadow"] > 0:
            out[f"{prefix}_shadow_iou"] = agg["shadow_iou"] / counts["shadow"]
            out[f"{prefix}_shadow_f1"] = agg["shadow_f1"] / counts["shadow"]
        if counts["reflection"] > 0:
            out[f"{prefix}_reflection_iou"] = agg["reflection_iou"] / counts["reflection"]
            out[f"{prefix}_reflection_f1"] = agg["reflection_f1"] / counts["reflection"]
            
        out["global_epoch"] = self.epoch_offset + epoch
        out["val_time_sec"] = time.perf_counter() - t0
        return out

    def fit(self, ckpt_path: str) -> None:
        if self.perform_baseline_eval:
            self.baseline_evaluation()
        
        best_f1 = 0.0
        patience_counter = 0
        
        ckpt_dir = os.path.dirname(ckpt_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_last = ckpt_path.replace('.pt', '_last.pt') if ckpt_path.endswith('.pt') else ckpt_path + '_last.pt'
        ckpt_best = ckpt_path.replace('.pt', '_best.pt') if ckpt_path.endswith('.pt') else ckpt_path + '_best.pt'
        
        for epoch in tqdm(range(1, self.max_epochs + 1)):
            train_stats = self.train_epoch(epoch)
            val_stats = self.validate(epoch)
            self.scheduler.step()
            
            current_global_epoch = self.epoch_offset + epoch
            sparsity = self.custom_sparsity if self.custom_sparsity is not None else self.calculate_sparsity()
            
            benchmark_speed_stats = self.benchmark(num_runs=100, warm_up=10)

            log_msg = (
                f"Epoch {epoch:03d} | F1: {val_stats['val_f1']:.4f} | "
                f"Loss: {train_stats['train_loss']:.4f}/{val_stats['val_loss']:.4f} | "
                f"Sparsity: {sparsity:.2%} | Time: {val_stats['val_time_sec']:.1f}s | "
                f"Inf: {benchmark_speed_stats['inference_ms']:.2f}ms"
            )
            log.info(log_msg)

            if self.wandb:
                log_dict = {
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "sparsity": sparsity,
                    **train_stats,
                    **val_stats,
                    **benchmark_speed_stats
                }
                self.wandb.log(log_dict, step=current_global_epoch)

            self.save_predictions(current_global_epoch)
            self.save_checkpoint(ckpt_last)

            if not math.isnan(val_stats["val_f1"]) and val_stats["val_f1"] > best_f1:
                best_f1 = val_stats["val_f1"]
                self.save_checkpoint(ckpt_best)
                log.info(f"New best F1={best_f1:.4f}. Saved: {ckpt_best}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log.info(f"Early Stopping triggered after {epoch} epochs (No improvement for {self.patience} epochs).")
                    break
                
        if self.qat_enabled:
            log.info("Converting QAT model to real Int8 quantized format...")
            
            self.model.to("cpu")
            quantize_(self.model, QATConfig(self.base_config, step="convert"))
            self.model.to(self.device)
            
            torch.save(self.model.state_dict(), ckpt_path.replace(".pt", "_int8_final.pt"))
            

    def save_checkpoint(self, path: str) -> None:
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(state, path)
        
    def calculate_sparsity(self) -> float:
        total_params = 0
        zero_params = 0
        
        for module in self.model.modules():
            if hasattr(module, "weight_mask"):
                mask = module.weight_mask
                total_params += mask.numel()
                zero_params += torch.sum(mask == 0).item()
                
            elif hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
                total_params += module.weight.numel()
        return 0.0 if total_params == 0 else zero_params / total_params