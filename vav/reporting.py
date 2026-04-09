from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vav.analysis import resize_heatmap


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_overlay_figure(
    image: Image.Image,
    heatmap: np.ndarray,
    output_path: str | Path,
    *,
    title: str,
    alpha: float = 0.45,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    resized_heatmap = resize_heatmap(heatmap, image.size)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(resized_heatmap, cmap="jet", alpha=alpha)
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_training_curves(training_csv: str | Path, output_path: str | Path) -> None:
    if not Path(training_csv).exists():
        print(f"Training CSV not found at {training_csv}, skipping training curves.")
        return
    rows = load_csv_rows(training_csv)
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) if row["train_loss"] else np.nan for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    val_acc = [float(row["val_acc"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="train_loss")
    axes[0].plot(epochs, val_loss, marker="o", label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Training and validation loss")

    axes[1].plot(epochs, val_acc, marker="o", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation accuracy")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_curves(analysis_csv: str | Path, output_path: str | Path) -> None:
    if not Path(analysis_csv).exists():
        print(f"Analysis CSV not found at {analysis_csv}, skipping metric curves.")
        return
    rows = load_csv_rows(analysis_csv)
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        layer_value = row["layer"]
        layer = -1 if layer_value == "" else int(layer_value)
        head = row.get("head", "")
        grouped[(row["mode"], layer, head)].append(row)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metric_names = ["entropy", "topk_concentration", "bbox_overlap"]
    for axis, metric_name in zip(axes, metric_names):
        for (mode, layer, head), records in grouped.items():
            by_epoch: dict[int, list[float]] = defaultdict(list)
            for record in records:
                value_text = record[metric_name]
                if value_text == "" or value_text.lower() == "nan":
                    continue
                by_epoch[int(record["checkpoint_epoch"])].append(float(value_text))
            if not by_epoch:
                continue
            epochs = sorted(by_epoch)
            means = [sum(by_epoch[epoch]) / len(by_epoch[epoch]) for epoch in epochs]
            if mode == "cls":
                head_text = "avg" if head == "" else head
                label = f"{mode} (layer {layer}, head {head_text})"
            else:
                label = mode
            axis.plot(epochs, means, marker="o", label=label)
        axis.set_title(metric_name.replace("_", " ").title())
        axis.set_xlabel("Checkpoint epoch")
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(loc="best", fontsize=8)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_checkpoint_gallery(overlay_dir: str | Path, output_path: str | Path ,mode: str = "rollout") -> None:
    if not Path(overlay_dir).exists():
        print(f"Overlay directory not found at {overlay_dir}, skipping checkpoint gallery.")
        return
    overlay_path = Path(overlay_dir)
    files = sorted(overlay_path.glob(f"*{mode}.png"))
    if not files:
        print(f"No {mode} images found in {overlay_dir}, skipping checkpoint gallery.")
        return
    sample_groups: dict[str, list[Path]] = defaultdict(list)
    for file_path in files:
        sample_key = file_path.stem.split("__")[0]
        sample_groups[sample_key].append(file_path)

    first_group = next(iter(sample_groups.values()))
    columns = len(first_group)
    rows = len(sample_groups)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    if rows == 1 and columns == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif columns == 1:
        axes = np.array([[axis] for axis in axes])

    for row_index, (_, group_files) in enumerate(sorted(sample_groups.items())):
        for col_index, file_path in enumerate(sorted(group_files)):
            axis = axes[row_index][col_index]
            axis.imshow(Image.open(file_path))
            axis.set_title(file_path.stem.replace("__", "\n"), fontsize=8)
            axis.axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_report_figures(
    *,
    training_csv: str | Path,
    analysis_csv: str | Path,
    overlay_dir: str | Path,
    output_dir: str | Path,
    
) -> None:
    figure_dir = Path(output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(training_csv, figure_dir / "training_curves.png")
    plot_metric_curves(analysis_csv, figure_dir / "attention_metrics.png")
    build_checkpoint_gallery(overlay_dir, figure_dir / "checkpoint_gallery.png")
