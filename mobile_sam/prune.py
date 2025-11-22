from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Literal

from rich.logging import RichHandler
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from mobile_sam.build_sam import sam_model_registry

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False
        )
    ]
)

log = logging.getLogger(__name__)


def modules_to_prune(model: nn.Module, include_linear: bool) -> List[Tuple[nn.Module, str]]:
    targets: List[Tuple[nn.Module, str]] = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            targets.append((m, "weight"))
            if include_linear and m.bias is not None:
                targets.append((m, "bias"))
        if include_linear and isinstance(m, nn.Linear):
            targets.append((m, "weight"))
            if m.bias is not None:
                targets.append((m, "bias"))
    return targets


def remove_pruning_reparam(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if hasattr(m, "weight_mask"):
                prune.remove(m, "weight")
            if hasattr(m, "bias_mask"):
                prune.remove(m, "bias")


def sparsity_report(model: nn.Module) -> str:
    total = 0
    zero = 0
    lines: List[str] = []
    for name, p in model.named_parameters():
        if p is None:
            continue
        numel = p.numel()
        z = int((p == 0).sum().item())
        total += numel
        zero += z
        if "weight" in name:
            lines.append(
                f"{name:60s} | shape={tuple(p.shape)} | zeros={z}/{numel} ({100.0 * z / numel:.2f}%)"
            )
    overall = f"Overall sparsity: {zero}/{total} ({100.0 * zero / total:.2f}%)"
    return "\n".join(lines + [overall])


def apply_pruning(
    model: nn.Module,
    mode: Literal["global_l1_unstructured", "layer_l1_unstructured", "layer_ln_structured", "random_unstructured"],
    amount: float,
    include_linear: bool,
    structured_n: int,
    structured_dim: int,
) -> None:
    params = modules_to_prune(model, include_linear=include_linear)

    if mode == "global_l1_unstructured":
        prune.global_unstructured(
            params,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        return

    if mode == "layer_l1_unstructured":
        for m, name in params:
            prune.l1_unstructured(m, name=name, amount=amount)
        return

    if mode == "layer_ln_structured":
        for m, name in params:
            if name != "weight":
                continue
            if not isinstance(m, (nn.Conv2d, nn.Linear)):
                continue
            prune.ln_structured(m, name=name, amount=amount, n=structured_n, dim=structured_dim)
        return

    if mode == "random_unstructured":
        for m, name in params:
            prune.random_unstructured(m, name=name, amount=amount)
        return

    raise ValueError(f"Unsupported pruning mode: {mode}")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-checkpoint",
        type=Path,
        required=True,
        help="Path to MobileSAM checkpoint to prune.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store pruned checkpoints.",
    )
    p.add_argument(
        "--include-linear",
        action="store_true",
        help="Include Linear layers (and their biases for unstructured pruning).",
    )
    p.add_argument(
        "--structured-n",
        type=int,
        default=2,
        help="Norm degree for ln_structured.",
    )
    p.add_argument(
        "--structured-dim",
        type=int,
        default=0,
        choices=[0, 1],
        help="Dimension along which to prune in ln_structured.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="global_l1_unstructured",
        choices=[
            "global_l1_unstructured",
            "layer_l1_unstructured",
            "layer_ln_structured",
            "random_unstructured",
        ],
        help="Pruning strategy to apply.",
    )
    p.add_argument(
        "--amount",
        type=float,
        default=0.8,
        help="Sparsity level to prune to (e.g., 0.8 = 80%% weights pruned).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (e.g. 'cpu' or 'cuda').",
    )
    return p.parse_args()


def main() -> None:
    args = get_args()
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    amount = args.amount
    mode = args.mode
    log.info(f"Applying pruning mode={mode}, amount={amount}")

    model = sam_model_registry["vit_t"](checkpoint=args.input_checkpoint)
    model.to(device)
    model.eval()

    apply_pruning(
        model=model,
        mode=mode,
        amount=amount,
        include_linear=args.include_linear,
        structured_n=args.structured_n,
        structured_dim=args.structured_dim,
    )

    remove_pruning_reparam(model)
    log.info(sparsity_report(model))

    save_path = args.output_dir / f"mobilesam_vit_t_{amount:.2f}_{mode}_pruned.pt"
    torch.save(model.state_dict(), save_path)
    log.info(f"Saved pruned checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
