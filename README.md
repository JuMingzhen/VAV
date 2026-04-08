# VAV: ViT Attention Dynamics Toolkit

VAV is now a small research toolkit for a course project on the intersection of cognitive science and artificial intelligence.

The project studies a simple but useful question:

**When a Vision Transformer is fine-tuned on a fine-grained bird classification task, does its attention gradually shift from diffuse/global patterns toward more task-relevant local regions?**

This repository contains:

- the original exploration notebook: [demo.ipynb](/C:/Users/JMZ/Desktop/Mind&Machine/VAV/demo.ipynb)
- an engineering-ready Python package: `vav/`
- a CLI for training, checkpoint analysis, and figure generation
- a report outline with theoretical grounding: [docs/report_outline.md](/C:/Users/JMZ/Desktop/Mind&Machine/VAV/docs/report_outline.md)

## Project Structure

```text
VAV/
├── demo.ipynb
├── docs/
│   └── report_outline.md
├── tests/
│   └── test_analysis.py
├── vav/
│   ├── analysis.py
│   ├── cli.py
│   ├── data.py
│   ├── modeling.py
│   ├── reporting.py
│   └── training.py
├── fish.pdf
├── pyproject.toml
└── requirements.txt
```

## What The Toolkit Does

The package is organized around four capabilities:

1. `modeling`
   Loads a ViT checkpoint and extracts all layer attentions.
2. `training`
   Fine-tunes a pre-trained ViT on CUB-200-2011 and saves checkpoints.
3. `analysis`
   Computes `cls`, `rollout`, and `all_layers` heatmaps, plus:
   - attention entropy
   - top-k patch concentration
   - bbox overlap
4. `reporting`
   Exports overlay images and report-ready summary plots.

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

The local `vit/` directory is still supported. If you already downloaded `google/vit-base-patch16-224` into `vit/`, the toolkit can use it directly.

## Dataset Layout

The training pipeline expects the CUB-200-2011 dataset in one of these forms:

```text
path/to/CUB_200_2011/
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
├── bounding_boxes.txt
├── classes.txt
└── images/
```

or

```text
path/to/dataset_root/
└── CUB_200_2011/
    ├── images.txt
    └── ...
```

## CLI Usage

### 1. Train a ViT on CUB

```bash
vav train \
  --dataset-root path/to/CUB_200_2011 \
  --output-dir outputs/cub_run \
  --model-name-or-path vit \
  --epochs 5 \
  --batch-size 8
```

This will:

- save checkpoints under `outputs/cub_run/checkpoints/`
- write `training_metrics.csv` with an `epoch=0` baseline
- write `training_config.json`

By default the saved checkpoints include `epoch_000`, `epoch_001`, `epoch_003`, `epoch_005`, and `final`.

### 2. Analyze Attention Across Checkpoints

```bash
vav analyze-checkpoints \
  --dataset-root path/to/CUB_200_2011 \
  --checkpoints-dir outputs/cub_run/checkpoints \
  --output-dir outputs/cub_analysis \
  --split test \
  --num-probes 4
```

This will:

- compute attention metrics across checkpoints
- save overlay images to `outputs/cub_analysis/overlays/`
- write `analysis_metrics.csv`

Default analysis modes:

- `rollout`
- `all_layers`
- `cls` for layers `-1,0,5,11`

Default head analysis for `cls`:

- average over heads
- head `0`
- head `3`
- head `7`

### 3. Build Report Figures

```bash
vav make-figures \
  --training-csv outputs/cub_run/training_metrics.csv \
  --analysis-csv outputs/cub_analysis/analysis_metrics.csv \
  --overlay-dir outputs/cub_analysis/overlays \
  --output-dir outputs/report_figures
```

This generates:

- `training_curves.png`
- `attention_metrics.png`
- `checkpoint_gallery.png`

## Smoke-Test Friendly Example

If you want to first confirm the pipeline works before running full training, use a tiny subset:

```bash
vav train \
  --dataset-root path/to/CUB_200_2011 \
  --output-dir outputs/smoke_run \
  --epochs 1 \
  --batch-size 2 \
  --max-train-samples 16 \
  --max-eval-samples 8
```

Then:

```bash
vav analyze-checkpoints \
  --dataset-root path/to/CUB_200_2011 \
  --checkpoints-dir outputs/smoke_run/checkpoints \
  --output-dir outputs/smoke_analysis \
  --num-probes 2
```

## Course Framing

This project is intentionally framed as a **course project**, not a publication-oriented research claim.

The main deliverables are:

- a working training and analysis pipeline
- experimental figures
- a clear report with theoretical grounding

The recommended theory links are:

- ViT: Dosovitskiy et al. (2021)
- Attention rollout: Abnar and Zuidema (2020)
- Interpretability caution: Chefer et al. (2021)
- Representation differences: Raghu et al. (2021)
- Global precedence: Navon (1981)
- Feature Integration Theory: Treisman and Gelade (1980)

## Verification

Fast checks you can run after installing dependencies:

```bash
python -m compileall vav tests
pytest tests/test_analysis.py
vav --help
```
