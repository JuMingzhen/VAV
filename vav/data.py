from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class CUBSample:
    sample_id: int
    image_id: int
    image_path: Path
    label: int
    class_name: str
    is_train: bool
    bbox: Optional[tuple[float, float, float, float]]


def _read_space_delimited(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=" ")
        for row in reader:
            clean = [item for item in row if item]
            if clean:
                rows.append(clean)
    return rows


def resolve_cub_root(root: str | Path) -> Path:
    root_path = Path(root)
    if (root_path / "images.txt").exists():
        return root_path
    candidate = root_path / "CUB_200_2011"
    if (candidate / "images.txt").exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find CUB metadata under {root_path}. Expected images.txt directly or in CUB_200_2011/."
    )


class CUB200Dataset(Dataset[CUBSample]):
    def __init__(self, root: str | Path, split: str = "train", transform: Optional[Callable[[Image.Image], object]] = None):
        self.root = resolve_cub_root(root)
        self.split = split
        self.transform = transform

        image_rows = _read_space_delimited(self.root / "images.txt")
        label_rows = _read_space_delimited(self.root / "image_class_labels.txt")
        split_rows = _read_space_delimited(self.root / "train_test_split.txt")
        bbox_rows = _read_space_delimited(self.root / "bounding_boxes.txt")
        class_rows = _read_space_delimited(self.root / "classes.txt")

        image_lookup = {int(image_id): rel_path for image_id, rel_path in image_rows}
        label_lookup = {int(image_id): int(class_id) - 1 for image_id, class_id in label_rows}
        split_lookup = {int(image_id): bool(int(is_train)) for image_id, is_train in split_rows}
        bbox_lookup = {
            int(image_id): (float(x), float(y), float(w), float(h))
            for image_id, x, y, w, h in bbox_rows
        }
        class_lookup = {int(class_id) - 1: class_name.replace(".", " ") for class_id, class_name in class_rows}

        if split not in {"train", "test", "all"}:
            raise ValueError("split must be one of: train, test, all")

        self.class_names = [class_lookup[idx] for idx in sorted(class_lookup)]
        self.samples: list[CUBSample] = []
        for sample_index, image_id in enumerate(sorted(image_lookup)):
            is_train = split_lookup[image_id]
            if split == "train" and not is_train:
                continue
            if split == "test" and is_train:
                continue
            label = label_lookup[image_id]
            image_path = self.root / "images" / image_lookup[image_id]
            self.samples.append(
                CUBSample(
                    sample_id=sample_index,
                    image_id=image_id,
                    image_path=image_path,
                    label=label,
                    class_name=class_lookup[label],
                    is_train=is_train,
                    bbox=bbox_lookup.get(image_id),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        payload: dict[str, object] = {"image": image, "sample": sample}
        if self.transform is not None:
            payload["image"] = self.transform(image)
        return payload
