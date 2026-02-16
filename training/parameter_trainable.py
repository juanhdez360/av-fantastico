import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


def freeze_all(model: nn.Module) -> None:
    """Disables requires_grad for all model parameters."""
    for p in model.parameters():
        p.requires_grad = False


def add_lora_to_action_head(model: nn.Module) -> nn.Module:
    """
    Freezes the model and adds LoRA adapters to the action head.
    Adjust `target_modules` to match the actual names in your model (e.g., 'action_proj', 'action_head', etc.).
    """
    freeze_all(model)
    target_modules = [
        "action_proj",
        "action_head",
        "policy_head",
        "actions_proj",
    ]
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def huber_loss(pred: torch.Tensor, gt: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss between prediction and ground truth."""
    return F.huber_loss(pred, gt, delta=delta)
