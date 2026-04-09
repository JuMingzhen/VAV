from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vav.analysis import resize_heatmap


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _checkpoint_sort_key(name: str) -> tuple[int, str]:
    if name == "final":
        return (10**9, name)
    if name.startswith("epoch_"):
        try:
            return (int(name.split("_")[-1]), name)
        except ValueError:
            return (10**8, name)
    return (10**7, name)


def _checkpoint_to_epoch(checkpoint_name: str) -> Optional[int]:
    if checkpoint_name.startswith("epoch_"):
        try:
            return int(checkpoint_name.split("_")[-1])
        except ValueError:
            return None
    return None


def _parse_overlay_file(file_path: Path) -> Optional[dict[str, object]]:
    parts = file_path.stem.split("__")
    if len(parts) < 3:
        return None

    sample_key = parts[0]
    checkpoint = parts[1]
    spec = "__".join(parts[2:])
    match = re.fullmatch(r"(?P<mode>[a-z_]+)(?:_layer_(?P<layer>-?\d+))?(?:_head_(?P<head>-?\d+))?", spec)
    if not match:
        return None

    mode = match.group("mode")
    layer_text = match.group("layer")
    head_text = match.group("head")
    return {
        "sample_key": sample_key,
        "checkpoint": checkpoint,
        "mode": mode,
        "layer": None if layer_text is None else int(layer_text),
        "head": None if head_text is None else int(head_text),
        "file_path": file_path,
    }


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


def build_checkpoint_gallery(
    overlay_dir: str | Path,
    output_path: str | Path,
    *,
    mode: str = "rollout",
    cls_layer: Optional[int] = None,
    cls_head: Optional[int] = None,
    epochs: Optional[list[int]] = None,
) -> None:
    if not Path(overlay_dir).exists():
        print(f"Overlay directory not found at {overlay_dir}, skipping checkpoint gallery.")
        return
    overlay_path = Path(overlay_dir)
    all_files = sorted(overlay_path.glob("*.png"))
    parsed = [_parse_overlay_file(path) for path in all_files]
    parsed = [item for item in parsed if item is not None]
    if not parsed:
        print(f"No valid overlay images found in {overlay_dir}, skipping checkpoint gallery.")
        return

    selected_epochs = None if epochs is None else set(epochs)
    filtered: list[dict[str, object]] = []
    for item in parsed:
        item_mode = item["mode"]
        if item_mode != mode:
            continue
        if mode == "cls":
            item_layer = item["layer"]
            item_head = item["head"]
            if cls_layer is not None and item_layer != cls_layer:
                continue
            if cls_head is not None and item_head != cls_head:
                continue

        if selected_epochs is not None:
            epoch_value = _checkpoint_to_epoch(str(item["checkpoint"]))
            if epoch_value is None or epoch_value not in selected_epochs:
                continue

        filtered.append(item)

    if not filtered:
        print("No overlays matched gallery filters, skipping checkpoint gallery.")
        return

    sample_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in filtered:
        sample_groups[str(item["sample_key"])].append(item)

    # Use a stable checkpoint column order shared by every sample row.
    checkpoint_names = sorted(
        {
            str(item["checkpoint"])
            for group in sample_groups.values()
            for item in group
        },
        key=_checkpoint_sort_key,
    )
    if not checkpoint_names:
        print("No checkpoint names could be parsed, skipping checkpoint gallery.")
        return

    columns = len(checkpoint_names) + 1  # 1st column is shared input image
    rows = len(sample_groups)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    if rows == 1 and columns == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif columns == 1:
        axes = np.array([[axis] for axis in axes])

    for row_index, (sample_key, group_items) in enumerate(sorted(sample_groups.items())):
        checkpoint_to_file: dict[str, Path] = {}
        for item in group_items:
            checkpoint_to_file[str(item["checkpoint"])] = Path(str(item["file_path"]))

        first_checkpoint = checkpoint_names[0]
        first_file = checkpoint_to_file.get(first_checkpoint)
        if first_file is None:
            first_file = next(iter(checkpoint_to_file.values()))

        with Image.open(first_file) as panel_image:
            width, height = panel_image.size
            split = width // 2
            input_panel = panel_image.crop((0, 0, split, height))

        input_axis = axes[row_index][0]
        input_axis.imshow(input_panel)
        input_axis.set_title(f"{sample_key}\nInput", fontsize=8)
        input_axis.axis("off")

        for col_index, checkpoint_name in enumerate(checkpoint_names, start=1):
            axis = axes[row_index][col_index]
            file_path = checkpoint_to_file.get(checkpoint_name)
            if file_path is None:
                axis.text(0.5, 0.5, "N/A", ha="center", va="center")
                axis.set_title(checkpoint_name, fontsize=8)
                axis.axis("off")
                continue

            with Image.open(file_path) as panel_image:
                width, height = panel_image.size
                split = width // 2
                heatmap_panel = panel_image.crop((split, 0, width, height))
            axis.imshow(heatmap_panel)
            axis.set_title(checkpoint_name, fontsize=8)
            axis.axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_report_figures(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    type: str = "gallery",
    gallery_mode: str = "rollout",
    gallery_cls_layer: Optional[int] = None,
    gallery_cls_head: Optional[int] = None,
    gallery_epochs: Optional[list[int]] = None,
) -> None:
    figure_dir = Path(output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if type == "training_curves":
        plot_training_curves(input_dir, figure_dir / "training_curves.png")
    elif type == "attention_metrics":
        plot_metric_curves(input_dir, figure_dir / "attention_metrics.png")
    elif type == "gallery":
        build_checkpoint_gallery(
            input_dir,
            figure_dir / "checkpoint_gallery.png",
            mode=gallery_mode,
            cls_layer=gallery_cls_layer,
            cls_head=gallery_cls_head,
            epochs=gallery_epochs,
        )
    else:
        print(f"Unknown report figure type: {type}, skipping.")