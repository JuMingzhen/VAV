from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _is_local_path(value: str) -> bool:
    return Path(value).exists()


@dataclass
class ModelOutputs:
    logits: torch.Tensor
    probabilities: torch.Tensor
    topk: list[tuple[int, float, str]]
    attentions: dict[int, np.ndarray]


class ViTAttentionModel:
    """Thin wrapper around Hugging Face ViT with attention extraction helpers."""

    def __init__(
        self,
        model_name_or_path: str = "vit",
        *,
        num_labels: Optional[int] = None,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        device: Optional[str] = None,
        local_files_only: Optional[bool] = None,
    ) -> None:
        model_path = str(model_name_or_path)
        local_only = _is_local_path(model_path) if local_files_only is None else local_files_only
        self.device = torch.device(device) if device else default_device()
        self.processor = ViTImageProcessor.from_pretrained(model_path, local_files_only=local_only)
        extra_kwargs = {}
        if num_labels is not None:
            extra_kwargs.update(
                {
                    "num_labels": num_labels,
                    "id2label": id2label or {i: str(i) for i in range(num_labels)},
                    "label2id": label2id or {str(i): i for i in range(num_labels)},
                    "ignore_mismatched_sizes": True,
                }
            )
        self.model = ViTForImageClassification.from_pretrained(
            model_path,
            local_files_only=local_only,
            attn_implementation="eager",
            **extra_kwargs,
        )
        self.model.config.output_attentions = True
        self.model.to(self.device)
        self.model.eval()

        cfg = self.model.config
        self.num_heads = cfg.num_attention_heads
        self.num_layers = cfg.num_hidden_layers
        patch_size = cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0]
        image_size = cfg.image_size if isinstance(cfg.image_size, int) else cfg.image_size[0]
        self.patch_size = patch_size
        self.image_size = image_size
        self.grid_size = image_size // patch_size
        self.id2label = {int(k): v for k, v in (getattr(cfg, "id2label", {}) or {}).items()}

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)

    def prepare_inputs(self, image: Image.Image) -> dict[str, torch.Tensor]:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        return {key: value.to(self.device) for key, value in inputs.items()}

    @torch.no_grad()
    def run(self, image: Image.Image, topk: int = 5) -> ModelOutputs:
        inputs = self.prepare_inputs(image)
        outputs = self.model(**inputs, output_attentions=True, return_dict=True)
        attentions = outputs.attentions
        if not attentions:
            raise RuntimeError("Model returned no attention tensors; output_attentions=True may be unsupported.")

        logits = outputs.logits[0].detach().cpu()
        probabilities = torch.softmax(logits, dim=0)
        topv, topi = probabilities.topk(min(topk, probabilities.shape[0]))
        top_predictions = []
        for idx_tensor, prob_tensor in zip(topi, topv):
            index = int(idx_tensor.item())
            label = self.id2label.get(index, f"class_{index}")
            top_predictions.append((index, float(prob_tensor.item()), label))

        attention_map = {layer_idx: attn[0].detach().cpu().numpy() for layer_idx, attn in enumerate(attentions)}
        return ModelOutputs(logits=logits, probabilities=probabilities, topk=top_predictions, attentions=attention_map)
