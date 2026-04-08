from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vav.analysis import AttentionAnalyzer, attention_entropy, bbox_overlap, topk_patch_concentration
from vav.data import CUB200Dataset
from vav.modeling import ViTAttentionModel
from vav.reporting import make_report_figures, save_overlay_figure
from vav.training import TrainingConfig, train_model


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_head_list(text: str) -> list[int | None]:
    heads: list[int | None] = []
    for item in text.split(","):
        value = item.strip().lower()
        if not value:
            continue
        if value in {"avg", "mean", "none"}:
            heads.append(None)
        else:
            heads.append(int(value))
    return heads


def _discover_checkpoints(checkpoints_dir: Path) -> list[Path]:
    candidates = [path for path in checkpoints_dir.iterdir() if path.is_dir()]
    epoch_candidates = [path for path in candidates if path.name.startswith("epoch_")]
    if epoch_candidates:
        candidates = epoch_candidates

    def sort_key(path: Path) -> tuple[int, str]:
        try:
            return (int(path.name.split("_")[-1]), path.name)
        except ValueError:
            return (10**8, path.name)

    return sorted(candidates, key=sort_key)


def _checkpoint_epoch(path: Path) -> int:
    if path.name == "final":
        return -1
    try:
        return int(path.name.split("_")[-1])
    except ValueError:
        return -1


def run_train(args: argparse.Namespace) -> None:
    config = TrainingConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        checkpoint_epochs=tuple(args.checkpoint_epochs),
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )
    output_dir = train_model(config)
    print(f"Training finished. Artifacts saved to {output_dir}")


def run_analyze_checkpoints(args: argparse.Namespace) -> None:
    dataset = CUB200Dataset(args.dataset_root, split=args.split)
    sample_indices = args.sample_indices or list(range(min(args.num_probes, len(dataset))))
    checkpoints = _discover_checkpoints(Path(args.checkpoints_dir))
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    layers = args.layers if args.layers else [-1]
    heads = args.heads if args.heads else [None]
    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "analysis_metrics.csv"

    rows: list[dict[str, object]] = []
    for checkpoint_dir in checkpoints:
        model = ViTAttentionModel(str(checkpoint_dir), device=args.device)
        analyzer = AttentionAnalyzer(model)
        checkpoint_epoch = _checkpoint_epoch(checkpoint_dir)
        for sample_index in sample_indices:
            item = dataset[sample_index]
            image = item["image"]
            sample = item["sample"]
            outputs = model.run(image)
            for mode in modes:
                target_layers = layers if mode == "cls" else [None]
                target_heads = heads if mode == "cls" else [None]
                for layer in target_layers:
                    for head in target_heads:
                        target_layer = -1 if layer is None else layer
                        heatmap = analyzer.compute_heatmap(outputs, mode=mode, layer=target_layer, head=head)
                        row = {
                            "checkpoint_name": checkpoint_dir.name,
                            "checkpoint_epoch": checkpoint_epoch,
                            "sample_index": sample_index,
                            "image_id": sample.image_id,
                            "class_name": sample.class_name,
                            "mode": mode,
                            "layer": "" if layer is None else layer,
                            "head": "" if head is None else head,
                            "pred_index": outputs.topk[0][0],
                            "pred_label": outputs.topk[0][2],
                            "pred_prob": round(outputs.topk[0][1], 6),
                            "is_correct": int(outputs.topk[0][0] == sample.label),
                            "entropy": round(attention_entropy(heatmap), 6),
                            "topk_concentration": round(topk_patch_concentration(heatmap, k=args.topk_patches), 6),
                            "bbox_overlap": round(bbox_overlap(heatmap, sample.bbox, image.size), 6),
                        }
                        rows.append(row)
                        if layer is None:
                            title = f"{checkpoint_dir.name} | {mode}"
                        else:
                            head_text = "avg" if head is None else str(head)
                            title = f"{checkpoint_dir.name} | cls layer {layer}, head {head_text}"
                        file_name = f"sample_{sample.image_id}__{checkpoint_dir.name}__{mode}"
                        if layer is not None and mode == "cls":
                            file_name += f"_layer_{layer}"
                        if head is not None and mode == "cls":
                            file_name += f"_head_{head}"
                        save_overlay_figure(image, heatmap, overlay_dir / f"{file_name}.png", title=title)

    if rows:
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "analysis_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_root": args.dataset_root,
                "split": args.split,
                "sample_indices": sample_indices,
                "modes": modes,
                "layers": layers,
                "heads": heads,
                "checkpoints_dir": args.checkpoints_dir,
            },
            handle,
            indent=2,
        )
    print(f"Analysis finished. Metrics saved to {metrics_path}")


def run_make_figures(args: argparse.Namespace) -> None:
    make_report_figures(
        training_csv=args.training_csv,
        analysis_csv=args.analysis_csv,
        overlay_dir=args.overlay_dir,
        output_dir=args.output_dir,
    )
    print(f"Figures saved to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VAV: ViT attention dynamics toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Fine-tune ViT on CUB-200-2011")
    train_parser.add_argument("--dataset-root", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--model-name-or-path", default="vit")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=5e-5)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--max-train-samples", type=int, default=None)
    train_parser.add_argument("--max-eval-samples", type=int, default=None)
    train_parser.add_argument("--checkpoint-epochs", type=_parse_int_list, default=[0, 1, 3, 5])
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--use-wandb", action="store_true")
    train_parser.add_argument("--wandb-project", default="vav-cub-training")
    train_parser.add_argument("--wandb-run-name", default=None)
    train_parser.add_argument("--wandb-entity", default=None)
    train_parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    train_parser.set_defaults(func=run_train)

    analyze_parser = subparsers.add_parser("analyze-checkpoints", help="Analyze attention across saved checkpoints")
    analyze_parser.add_argument("--dataset-root", required=True)
    analyze_parser.add_argument("--checkpoints-dir", required=True)
    analyze_parser.add_argument("--output-dir", required=True)
    analyze_parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    analyze_parser.add_argument("--num-probes", type=int, default=4)
    analyze_parser.add_argument("--sample-indices", type=_parse_int_list, default=None)
    analyze_parser.add_argument("--modes", default="rollout,all_layers,cls")
    analyze_parser.add_argument("--layers", type=_parse_int_list, default=[-1, 0, 5, 11])
    analyze_parser.add_argument("--heads", type=_parse_head_list, default=[None, 0, 3, 7])
    analyze_parser.add_argument("--topk-patches", type=int, default=10)
    analyze_parser.add_argument("--device", default=None)
    analyze_parser.set_defaults(func=run_analyze_checkpoints)

    figures_parser = subparsers.add_parser("make-figures", help="Build report-ready figures from CSV outputs")
    figures_parser.add_argument("--training-csv", required=True)
    figures_parser.add_argument("--analysis-csv", required=True)
    figures_parser.add_argument("--overlay-dir", required=True)
    figures_parser.add_argument("--output-dir", required=True)
    figures_parser.set_defaults(func=run_make_figures)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
