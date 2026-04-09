# VAV: ViT Attention Visualization Toolkit

VAV is a small research toolkit for probing into attention machanism of ViT (Vision Transformer).
This is a project of NUS course: NST2062 Mind and Machine.

The project studies a simple but useful question:

**When a Vision Transformer is fine-tuned on a fine-grained bird classification task, does its attention gradually shift from diffuse/global patterns toward more task-relevant local regions? Are there any patterns connected with cognitive science?**

This repository contains:

- the original exploration notebook: [demo.ipynb]
- an engineering-ready Python package: `vav/`
- a CLI for training, checkpoint analysis, and report figure generation
- a report outline with theoretical grounding: [docs/report_outline.md]

## Project Structure

```text
VAV/
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
├── demo/
|   └── demo.ipynb
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

## Dataset Layout

You can download the dataset in [https://data.caltech.edu/records/65de6-vp158].

The training pipeline expects the CUB-200-2011 dataset in form:
```text
path/to/CUB_200_2011/
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
├── bounding_boxes.txt
├── classes.txt
└── images/
```

## Model

The toolkit supports all models wrapped by the Transformers library. 
For this experiment, the following is recommended: [https://huggingface.co/google/vit-base-patch16-224-in21k].

You can download it via this command (in terminal): 
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/vit-base-patch16-224-in21k, local_dir='./model')"
```

## CLI Usage

### 1. Train a ViT on CUB

```bash
vav train \
  --dataset-root CUB_200_2011 \
  --output-dir outputs/cub_run \
  --model-name-or-path vit \
  --epochs 5 \
  --batch-size 8
```

This will:

- save checkpoints under `outputs/cub_run/checkpoints/`
- write `training_metrics.csv` with an `epoch=0` baseline
- write `training_config.json`
- record training process and upload to 'wandb' if turned on in args

By default the saved checkpoints include `epoch_000`, `epoch_001`, `epoch_003`, `epoch_005`, and `final`.

### 2. Analyze Attention Across Checkpoints

```bash
vav analyze-checkpoints \
  --dataset-root CUB_200_2011 \
  --checkpoints-dir outputs/cub_run/checkpoints \
  --output-dir outputs/cub_analysis \
  --split test \
  --num-probes 4
```

This will:

- compute attention metrics across checkpoints
- save overlay images (heatmaps) to `outputs/cub_analysis/overlays/`
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
