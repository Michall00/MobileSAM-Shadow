from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path
from typing import List, tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm import tqdm

from mobile_sam.build_sam import sam_model_registry


class SparseBSRLinear(nn.Module):
    def __init__(self, dense: nn.Linear, blocksize: int = 4) -> None:
        super().__init__()
        self.in_features = dense.in_features
        self.out_features = dense.out_features
        self.blocksize = blocksize

        weight_t = dense.weight.detach().t()  # (in_features, out_features)
        self.register_buffer("weight", weight_t.to_sparse_bsr((blocksize, blocksize)))

        if dense.bias is not None:
            self.register_buffer("bias", dense.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        y_flat = torch.matmul(x_flat, self.weight)
        if self.bias is not None:
            y_flat = y_flat + self.bias
        return y_flat.reshape(*orig_shape[:-1], self.out_features)


class SparseLinear(nn.Module):
    def __init__(self, dense: nn.Linear) -> None:
        super().__init__()
        self.in_features = dense.in_features
        self.out_features = dense.out_features

        weight_t = dense.weight.detach().t()  # (in_features, out_features)
        self.register_buffer("weight", weight_t.to_sparse_csr())

        if dense.bias is not None:
            self.register_buffer("bias", dense.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)  # (..., in_features)
        y_flat = torch.matmul(x_flat, self.weight)  # (..., out_features)
        if self.bias is not None:
            y_flat = y_flat + self.bias
        y = y_flat.reshape(*orig_shape[:-1], self.out_features)
        return y


def modules_to_prune(model: nn.Module, include_linear: bool) -> list[tuple[nn.Module, str]]:
    targets: list[tuple[nn.Module, str]] = []
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
    lines: list[str] = []
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
    mode: str,
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


def replace_linears_with_sparse_variants(
    module: nn.Module,
    use_csr: bool = False,
    use_bsr: bool = False,
    blocksize: int = 4,
) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if use_csr:
                setattr(module, name, SparseLinear(child))
            elif use_bsr:
                setattr(module, name, SparseBSRLinear(child, blocksize=blocksize))
        else:
            replace_linears_with_sparse_variants(child, use_csr, use_bsr, blocksize)
    return module


def count_dense_and_sparse_linears(module: nn.Module) -> None:
    n_linear = 0
    n_sparse_csr = 0
    n_sparse_bsr = 0

    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            n_linear += 1
            print(f"[DENSE]  {name}: in={m.in_features}, out={m.out_features}")
        elif isinstance(m, SparseLinear):
            n_sparse_csr += 1
            print(f"[CSR]    {name}: in={m.in_features}, out={m.out_features}")
        elif isinstance(m, SparseBSRLinear):
            n_sparse_bsr += 1
            print(f"[BSR]    {name}: in={m.in_features}, out={m.out_features}")

    print("\nSummary:")
    print(f"  nn.Linear      : {n_linear}")
    print(f"  SparseLinear   : {n_sparse_csr}")
    print(f"  SparseBSRLinear: {n_sparse_bsr}")


@torch.no_grad()
def benchmark_encoder(
    encoder: nn.Module,
    device: str,
    input_shape: tuple[int, int, int, int] = (1, 3, 1024, 1024),
    iters: int = 10,
) -> float:
    encoder.eval()
    dev = torch.device(device)
    dummy = torch.randn(*input_shape, device=dev)

    for _ in range(3):
        _ = encoder(dummy)

    start = time.perf_counter()
    for _ in tqdm(range(iters)):
        _ = encoder(dummy)
    end = time.perf_counter()
    return (end - start) / iters


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
        default=0.5,
        help="Sparsity level to prune to (e.g., 0.5 = 50%% weights pruned).",
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
    print(f"\n[INFO] Applying pruning mode={mode}, amount={amount}")

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
    print(sparsity_report(model))

    if mode == "global_l1_unstructured":
        print("[INFO] Benchmarking image_encoder dense vs sparse-linear")

        encoder_dense = deepcopy(model.image_encoder).to(device)

        encoder_csr = deepcopy(model.image_encoder).to(device)
        replace_linears_with_sparse_variants(encoder_csr, use_csr=True)

        encoder_bsr = deepcopy(model.image_encoder).to(device)
        replace_linears_with_sparse_variants(encoder_bsr, use_bsr=True, blocksize=4)

        t_dense = benchmark_encoder(encoder_dense, device=device)
        t_csr = benchmark_encoder(encoder_csr, device=device)
        # t_bsr = benchmark_encoder(encoder_bsr, device=device)

        print("[DENSE ENCODER]")
        count_dense_and_sparse_linears(encoder_dense)

        print("\n[SPARSE CSR ENCODER]")
        count_dense_and_sparse_linears(encoder_csr)

        print("\n[SPARSE BSR ENCODER]")
        count_dense_and_sparse_linears(encoder_bsr)

        print(f"Dense: {t_dense * 1000:.3f} ms")
        print(f"CSR:   {t_csr * 1000:.3f} ms")
        # print(f"BSR:   {t_bsr*1000:.3f} ms")

    save_path = args.output_dir / f"mobilesam_vit_t_{amount:.2f}_{mode}_pruned.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Saved pruned checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
