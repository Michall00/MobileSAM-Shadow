from torch import nn
from mobile_sam.build_sam import sam_model_registry


def load_mobilesam_vit_t(ckpt_path: str | None, device: str = "cuda") -> nn.Module:
    model = sam_model_registry["vit_t"](checkpoint=ckpt_path)
    model.to(device)
    for p in model.prompt_encoder.parameters():
        p.requires_grad = False
    for p in model.mask_decoder.parameters():
        p.requires_grad = False
    for p in model.image_encoder.parameters():
        p.requires_grad = True
    model.train()
    return model


def freeze_non_encoder(model: nn.Module) -> list[nn.Parameter]:
    for p in getattr(model, "prompt_encoder").parameters():
        p.requires_grad = False
    for p in getattr(model, "mask_decoder").parameters():
        p.requires_grad = False
    enc_params = [p for p in getattr(model, "image_encoder").parameters()]
    for p in enc_params:
        p.requires_grad = True
    return enc_params
