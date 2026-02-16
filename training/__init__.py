"""Training package: dataset, LoRA/utils, and training loop."""

from .dataset import ClipsDataset
from .parameter_trainable import add_lora_to_action_head, freeze_all, huber_loss
from .training import train

__all__ = [
    "ClipsDataset",
    "add_lora_to_action_head",
    "freeze_all",
    "huber_loss",
    "train",
]
