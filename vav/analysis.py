from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from vav.modeling import ModelOutputs, ViTAttentionModel


def normalize_map(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    minimum = float(arr.min())
    maximum = float(arr.max())
    return (arr - minimum) / (maximum - minimum + 1e-8)


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    scaled = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8))
    resized = scaled.resize(size, Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def attention_entropy(heatmap: np.ndarray) -> float:
    arr = np.asarray(heatmap, dtype=np.float64)
    flat = arr.reshape(-1)
    total = float(flat.sum())
    if total <= 0:
        return 0.0
    probs = flat / total
    return float(-(probs * np.log(probs + 1e-12)).sum())


def topk_patch_concentration(heatmap: np.ndarray, k: int = 10) -> float:
    arr = np.asarray(heatmap, dtype=np.float64).reshape(-1)
    total = float(arr.sum())
    if total <= 0:
        return 0.0
    topk = np.sort(arr)[-min(k, arr.size) :]
    return float(topk.sum() / total)


def bbox_overlap(heatmap: np.ndarray, bbox: Optional[tuple[float, float, float, float]], image_size: tuple[int, int]) -> float:
    if bbox is None:
        return float("nan")
    width, height = image_size
    if width <= 0 or height <= 0:
        return float("nan")
    x, y, w, h = bbox
    x0 = max(0, min(width, x))
    y0 = max(0, min(height, y))
    x1 = max(x0, min(width, x + w))
    y1 = max(y0, min(height, y + h))
    if x1 <= x0 or y1 <= y0:
        return 0.0

    hm = np.asarray(heatmap, dtype=np.float64)
    mask = np.zeros_like(hm, dtype=np.float64)
    rows, cols = hm.shape
    cx0 = int(np.floor((x0 / width) * cols))
    cy0 = int(np.floor((y0 / height) * rows))
    cx1 = int(np.ceil((x1 / width) * cols))
    cy1 = int(np.ceil((y1 / height) * rows))
    mask[max(0, cy0) : min(rows, cy1), max(0, cx0) : min(cols, cx1)] = 1.0
    total = float(hm.sum())
    if total <= 0:
        return 0.0
    return float((hm * mask).sum() / total)


@dataclass
class AttentionResult:
    heatmap: np.ndarray
    outputs: ModelOutputs


class AttentionAnalyzer:
    def __init__(self, model: ViTAttentionModel) -> None:
        self.model = model

    def analyze(self, image: Image.Image) -> AttentionResult:
        outputs = self.model.run(image)
        return AttentionResult(heatmap=self.rollout(outputs), outputs=outputs)

    def cls_attention(self, outputs: ModelOutputs, layer: int = -1, head: Optional[int] = None) -> np.ndarray:
        attention = outputs.attentions[layer % self.model.num_layers]
        row = attention[:, 0, 1:]
        weights = row.mean(0) if head is None else row[head]
        return normalize_map(weights.reshape(self.model.grid_size, self.model.grid_size))

    def rollout(self, outputs: ModelOutputs) -> np.ndarray:
        token_count = next(iter(outputs.attentions.values())).shape[-1]
        result = np.eye(token_count, dtype=np.float64)
        for layer_idx in range(self.model.num_layers):
            layer_attention = outputs.attentions[layer_idx].mean(0) + np.eye(token_count)
            layer_attention = layer_attention / (layer_attention.sum(axis=-1, keepdims=True) + 1e-8)
            result = layer_attention @ result
        return normalize_map(result[0, 1:].reshape(self.model.grid_size, self.model.grid_size))

    def all_layers_average(self, outputs: ModelOutputs) -> np.ndarray:
        maps = [self.cls_attention(outputs, layer=layer_idx, head=None) for layer_idx in range(self.model.num_layers)]
        return normalize_map(np.mean(maps, axis=0))

    def compute_heatmap(
        self,
        outputs: ModelOutputs,
        *,
        mode: str = "rollout",
        layer: int = -1,
        head: Optional[int] = None,
    ) -> np.ndarray:
        if mode == "rollout":
            return self.rollout(outputs)
        if mode == "all_layers":
            return self.all_layers_average(outputs)
        if mode == "cls":
            return self.cls_attention(outputs, layer=layer, head=head)
        raise ValueError(f"Unsupported mode: {mode}")
