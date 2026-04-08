from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor

from vav.data import CUB200Dataset
from vav.modeling import ViTAttentionModel


@dataclass
class TrainingConfig:
    dataset_root: str
    output_dir: str
    model_name_or_path: str = "vit"
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    num_workers: int = 0
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    checkpoint_epochs: tuple[int, ...] = (0, 1, 3, 5)
    device: Optional[str] = None
    use_wandb: bool = False
    wandb_project: str = "vav-cub-training"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"


def _slice_dataset(dataset: CUB200Dataset, limit: Optional[int]) -> CUB200Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    dataset.samples = dataset.samples[:limit]
    return dataset


def _build_collate_fn(processor: ViTImageProcessor):
    def collate(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        labels = torch.tensor([item["sample"].label for item in batch], dtype=torch.long)
        encoded = processor(images=images, return_tensors="pt")
        encoded["labels"] = labels
        return encoded

    return collate


def _evaluate(model: ViTAttentionModel, loader: DataLoader) -> tuple[float, float]:
    model.model.eval()
    loss_sum = 0.0
    count = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(model.device)
            labels = batch["labels"].to(model.device)
            outputs = model.model(pixel_values=pixel_values, labels=labels)
            loss_sum += float(outputs.loss.item()) * labels.shape[0]
            predictions = outputs.logits.argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            count += int(labels.shape[0])
    if count == 0:
        return 0.0, 0.0
    return loss_sum / count, correct / count


def _save_metrics(metrics_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _init_wandb(config: TrainingConfig, output_dir: Path):
    if not config.use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "Weights & Biases is not installed. Run `pip install wandb` before enabling wandb logging."
        ) from exc

    return wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        entity=config.wandb_entity,
        mode=config.wandb_mode,
        dir=str(output_dir),
        config=asdict(config),
    )


def train_model(config: TrainingConfig) -> Path:
    output_dir = Path(config.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(config, output_dir)

    train_dataset = _slice_dataset(CUB200Dataset(config.dataset_root, split="train"), config.max_train_samples)
    test_dataset = _slice_dataset(CUB200Dataset(config.dataset_root, split="test"), config.max_eval_samples)

    id2label = {idx: name for idx, name in enumerate(train_dataset.class_names)}
    label2id = {name: idx for idx, name in id2label.items()}
    model = ViTAttentionModel(
        config.model_name_or_path,
        num_labels=len(train_dataset.class_names),
        id2label=id2label,
        label2id=label2id,
        device=config.device,
    )

    collate_fn = _build_collate_fn(model.processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = AdamW(model.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    metrics_rows: list[dict[str, object]] = []
    save_epochs = set(config.checkpoint_epochs) | {config.epochs}

    initial_dir = checkpoints_dir / "epoch_000"
    model.save_pretrained(initial_dir)
    initial_val_loss, initial_val_acc = _evaluate(model, test_loader)
    initial_row = {
        "epoch": 0,
        "train_loss": "",
        "val_loss": round(initial_val_loss, 6),
        "val_acc": round(initial_val_acc, 6),
        "train_samples": len(train_dataset),
        "eval_samples": len(test_dataset),
    }
    metrics_rows.append(initial_row)
    if wandb_run is not None:
        wandb_run.log(
            {
                "epoch": 0,
                "eval/val_loss": initial_val_loss,
                "eval/val_acc": initial_val_acc,
                "dataset/train_samples": len(train_dataset),
                "dataset/eval_samples": len(test_dataset),
            },
            step=0,
        )

    global_step = 0
    try:
        for epoch in range(1, config.epochs + 1):
            model.model.train()
            train_loss_sum = 0.0
            train_count = 0
            for batch_index, batch in enumerate(train_loader, start=1):
                pixel_values = batch["pixel_values"].to(model.device)
                labels = batch["labels"].to(model.device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model.model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                batch_size = int(labels.shape[0])
                loss_value = float(loss.item())
                train_loss_sum += loss_value * batch_size
                train_count += batch_size
                global_step += 1

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/step_loss": loss_value,
                            "train/running_loss": train_loss_sum / max(train_count, 1),
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch,
                            "train/epoch_progress": epoch - 1 + (batch_index / max(len(train_loader), 1)),
                        },
                        step=global_step,
                    )

            train_loss = train_loss_sum / max(train_count, 1)
            val_loss, val_acc = _evaluate(model, test_loader)
            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "train_samples": len(train_dataset),
                "eval_samples": len(test_dataset),
            }
            metrics_rows.append(row)

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/epoch_loss": train_loss,
                        "eval/val_loss": val_loss,
                        "eval/val_acc": val_acc,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "dataset/train_samples": len(train_dataset),
                        "dataset/eval_samples": len(test_dataset),
                    },
                    step=global_step,
                )

            if epoch in save_epochs:
                model.save_pretrained(checkpoints_dir / f"epoch_{epoch:03d}")

        final_dir = checkpoints_dir / "final"
        model.save_pretrained(final_dir)

        _save_metrics(output_dir / "training_metrics.csv", metrics_rows)
        with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return output_dir
