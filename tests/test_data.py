from pathlib import Path

from PIL import Image

from vav.data import CUB200Dataset, resolve_cub_root


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_resolve_cub_root_supports_nested_layout(tmp_path):
    root = tmp_path / "dataset_root" / "CUB_200_2011"
    root.mkdir(parents=True)
    (root / "images" / "class_a").mkdir(parents=True)
    _write_lines(root / "images.txt", ["1 class_a/image_1.jpg"])
    _write_lines(root / "image_class_labels.txt", ["1 1"])
    _write_lines(root / "train_test_split.txt", ["1 1"])
    _write_lines(root / "bounding_boxes.txt", ["1 0 0 10 10"])
    _write_lines(root / "classes.txt", ["1 class_a"])
    Image.new("RGB", (16, 16), color="red").save(root / "images" / "class_a" / "image_1.jpg")

    assert resolve_cub_root(tmp_path / "dataset_root") == root


def test_cub_dataset_reads_sample_and_bbox(tmp_path):
    root = tmp_path / "CUB_200_2011"
    (root / "images" / "class_a").mkdir(parents=True)
    _write_lines(root / "images.txt", ["1 class_a/image_1.jpg", "2 class_a/image_2.jpg"])
    _write_lines(root / "image_class_labels.txt", ["1 1", "2 1"])
    _write_lines(root / "train_test_split.txt", ["1 1", "2 0"])
    _write_lines(root / "bounding_boxes.txt", ["1 1 2 10 12", "2 0 0 8 8"])
    _write_lines(root / "classes.txt", ["1 class_a"])
    Image.new("RGB", (16, 16), color="red").save(root / "images" / "class_a" / "image_1.jpg")
    Image.new("RGB", (16, 16), color="blue").save(root / "images" / "class_a" / "image_2.jpg")

    train_dataset = CUB200Dataset(root, split="train")
    test_dataset = CUB200Dataset(root, split="test")

    assert len(train_dataset) == 1
    assert len(test_dataset) == 1
    assert train_dataset[0]["sample"].bbox == (1.0, 2.0, 10.0, 12.0)
