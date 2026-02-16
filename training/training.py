import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from .dataset import ClipsDataset
    from .parameter_trainable import add_lora_to_action_head, huber_loss
except ImportError:
    from dataset import ClipsDataset
    from parameter_trainable import add_lora_to_action_head, huber_loss


def train(
    model,
    train_ds: ClipsDataset,
    *,
    device: str | None = None,
    batch_size: int = 2,
    num_workers: int = 2,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    use_lora: bool = True,
):
    """
    - model: modelo que acepta image_frames, ego_history_xyz, ego_history_rot
      y devuelve dict con "actions".
    - train_ds: ClipsDataset (cada muestra debe tener actions_gt).
    - use_lora: si True, congela el modelo y añade LoRA al head de acciones.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if use_lora:
        model = add_lora_to_action_head(model)
    model.train()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ClipsDataset.collate_fn,
        pin_memory=(device == "cuda"),
    )

    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            image_frames = batch["image_frames"].to(device)
            ego_xyz = batch["ego_history_xyz"].to(device)
            ego_rot = batch["ego_history_rot"].to(device)
            actions_gt = batch["actions_gt"].to(device)

            image_frames = image_frames.float() / 255.0

            out = model(
                image_frames=image_frames,
                ego_history_xyz=ego_xyz,
                ego_history_rot=ego_rot,
            )
            pred_actions = out["actions"]

            loss = huber_loss(pred_actions, actions_gt)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

            total_loss += loss.item()

        mean_loss = total_loss / len(train_loader)
        print(f"epoch {epoch} loss={mean_loss:.4f}")

    return model


if __name__ == "__main__":
    print("Uso: desde código, importa train() y pásale model y ClipsDataset.")
    print("  from src import train, ClipsDataset")
    print("  train(model, train_ds, epochs=10, batch_size=2)")
