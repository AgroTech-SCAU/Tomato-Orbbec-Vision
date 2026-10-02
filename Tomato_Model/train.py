"""Train the E13 YOLO11m-seg model for software V1.0.

The script contains the complete E13 training configuration and expects only:

    train.py
    yolo11m-seg.pt
    dataset/
        train/images, train/labels
        valid/images, valid/labels
        test/images,  test/labels

It validates the frozen Roboflow v18 dataset, generates ``data.yaml``, trains
the model, and records the core artifacts needed by the V1.0 software project.
ONNX, TensorRT, quantization, and formal deployment benchmarks are deliberately
outside this script and outside the V1.0 training scope.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset"
DATA_YAML = ROOT / "data.yaml"
WEIGHTS = ROOT / "yolo11m-seg.pt"
RUNS_DIR = ROOT / "runs" / "segment"

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

DEFAULT_RUN_NAME = "v18_e13_mosaic025"
EXPECTED_IMAGE_SIZE = (768, 768)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = {
    "train": {"directory": "train", "images": 763, "instances": {0: 6420, 1: 659, 2: 3515}},
    "val": {"directory": "valid", "images": 96, "instances": {0: 806, 1: 81, 2: 524}},
    "test": {"directory": "test", "images": 96, "instances": {0: 741, 1: 82, 2: 559}},
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_data_yaml() -> None:
    """Create the project-local dataset configuration from fixed content."""
    content = (
        "train: dataset/train/images\n"
        "val: dataset/valid/images\n"
        "test: dataset/test/images\n"
        "\n"
        "nc: 3\n"
        "names: [ripe, stem, unripe]\n"
        "\n"
        "roboflow:\n"
        "  workspace: buzhidao-mkdfs\n"
        "  project: tomato-h72eq\n"
        "  version: 18\n"
        "  license: CC BY 4.0\n"
        "  url: https://universe.roboflow.com/buzhidao-mkdfs/tomato-h72eq/dataset/18\n"
    )
    if not DATA_YAML.is_file() or DATA_YAML.read_text(encoding="utf-8") != content:
        DATA_YAML.write_text(content, encoding="utf-8", newline="\n")


def audit_split(split: str, directory: str) -> tuple[Counter[int], int]:
    """Validate one image/segmentation-label split before GPU allocation."""
    split_root = DATASET_ROOT / directory
    image_dir = split_root / "images"
    label_dir = split_root / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Missing images/labels directory for {split}: {split_root}"
        )

    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = sorted(label_dir.glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    problems: list[str] = []

    expected_images = int(SPLITS[split]["images"])
    if len(images) != expected_images or len(labels) != expected_images:
        problems.append(
            f"expected {expected_images} images and labels, found "
            f"{len(images)} images and {len(labels)} labels"
        )
    if image_stems != label_stems:
        problems.append(
            "unmatched files: "
            f"missing_labels={sorted(image_stems - label_stems)}, "
            f"orphan_labels={sorted(label_stems - image_stems)}"
        )

    for image_path in images:
        with Image.open(image_path) as image:
            if image.size != EXPECTED_IMAGE_SIZE:
                problems.append(
                    f"{image_path.name}: image size is {image.size}, expected {EXPECTED_IMAGE_SIZE}"
                )

    class_counts: Counter[int] = Counter()
    for label_path in labels:
        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if not lines:
            problems.append(
                f"{label_path.name}: empty annotation (must be reviewed, not assumed negative)"
            )
            continue

        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            try:
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except (ValueError, IndexError):
                problems.append(
                    f"{label_path.name}:{line_number}: non-numeric or missing fields"
                )
                continue

            if class_id not in (0, 1, 2):
                problems.append(f"{label_path.name}:{line_number}: invalid class {class_id}")
            else:
                class_counts[class_id] += 1

            if len(fields) == 5:
                problems.append(
                    f"{label_path.name}:{line_number}: bbox-only row; "
                    "a segmentation polygon is required"
                )
            elif len(coordinates) < 6 or len(coordinates) % 2 != 0:
                problems.append(
                    f"{label_path.name}:{line_number}: invalid segmentation polygon"
                )
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                problems.append(
                    f"{label_path.name}:{line_number}: coordinate outside [0, 1]"
                )

    expected_counts = Counter(SPLITS[split]["instances"])
    if class_counts != expected_counts:
        problems.append(
            f"instance counts differ from E13 v18: expected {dict(expected_counts)}, "
            f"found {dict(class_counts)}"
        )

    if problems:
        details = "\n  - ".join(problems)
        raise ValueError(f"Dataset audit failed for {split}:\n  - {details}")
    return class_counts, len(images)


def audit_dataset() -> dict[str, Any]:
    """Fail before training if local inputs are not the fixed E13 v18 data."""
    total_counts: Counter[int] = Counter()
    split_summary: dict[str, Any] = {}
    total_images = 0
    for split, expected in SPLITS.items():
        counts, image_count = audit_split(split, str(expected["directory"]))
        total_counts.update(counts)
        total_images += image_count
        split_summary[split] = {
            "images": image_count,
            "instances": {str(key): value for key, value in sorted(counts.items())},
        }

    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Missing pretrained weights: {WEIGHTS}")
    summary = {
        "checked_at": now_iso(),
        "status": "passed",
        "image_size": list(EXPECTED_IMAGE_SIZE),
        "total_images": total_images,
        "total_instances": sum(total_counts.values()),
        "class_instances": {str(key): value for key, value in sorted(total_counts.items())},
        "splits": split_summary,
    }
    print(
        f"Dataset audit passed: {total_images} images; "
        f"instances={dict(sorted(total_counts.items()))}"
    )
    return summary


def e13_training_arguments(run_name: str = DEFAULT_RUN_NAME) -> dict[str, Any]:
    """Return the fixed E13 training configuration."""
    return {
        "task": "segment",
        "data": str(DATA_YAML),
        "epochs": 200,
        "time": None,
        "patience": 40,
        "batch": 2,
        "imgsz": 768,
        "save": True,
        "save_period": -1,
        "cache": False,
        "device": "0",
        "workers": 0,
        "project": str(RUNS_DIR),
        "name": run_name,
        "exist_ok": False,
        "pretrained": True,
        "cls_remap": True,
        "optimizer": "SGD",
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "single_cls": False,
        "rect": False,
        "cos_lr": True,
        "close_mosaic": 30,
        "resume": False,
        "amp": True,
        "fraction": 1.0,
        "profile": False,
        "freeze": None,
        "multi_scale": 0.0,
        "compile": False,
        "overlap_mask": True,
        "mask_ratio": 2,
        "dropout": 0.0,
        "val": True,
        "split": "val",
        "save_json": False,
        "conf": None,
        "iou": 0.7,
        "max_det": 300,
        "quantize": None,
        "dnn": False,
        "plots": True,
        "end2end": None,
        "augment": False,
        "agnostic_nms": False,
        "classes": None,
        "retina_masks": False,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "distill_model": None,
        "dis": 6.0,
        "box": 7.5,
        "cls": 0.5,
        "cls_pw": 0.0,
        "dfl": 1.5,
        "pose": 12.0,
        "kobj": 1.0,
        "rle": 1.0,
        "angle": 1.0,
        "nbs": 64,
        "hsv_h": 0.01,
        "hsv_s": 0.4,
        "hsv_v": 0.3,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.25,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic": 0.25,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "copy_paste_mode": "flip",
        "auto_augment": "randaugment",
        "erasing": 0.4,
    }


def summarize_results_csv(path: Path, planned_epochs: int) -> dict[str, Any]:
    """Extract the values needed to finish the V1.0 training record."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {str(key).strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise RuntimeError(f"Training results are empty: {path}")

    for row in rows:
        row["_fitness"] = float(row["metrics/mAP50-95(B)"]) + float(
            row["metrics/mAP50-95(M)"]
        )
    best = max(rows, key=lambda row: float(row["_fitness"]))
    loss_keys = (
        "train/box_loss",
        "train/seg_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/seg_loss",
        "val/cls_loss",
        "val/dfl_loss",
    )
    return {
        "epochs_recorded": len(rows),
        "last_epoch": int(float(rows[-1]["epoch"])),
        "planned_epochs": planned_epochs,
        "early_stopped": len(rows) < planned_epochs,
        "best_epoch": int(float(best["epoch"])),
        "best_fitness": float(best["_fitness"]),
        "best_valid_box_map50_95": float(best["metrics/mAP50-95(B)"]),
        "best_valid_mask_map50_95": float(best["metrics/mAP50-95(M)"]),
        "minimum_losses": {
            key: min(float(row[key]) for row in rows) for key in loss_keys
        },
    }


def write_training_record(
    run_dir: Path,
    config: dict[str, Any],
    dataset_audit: dict[str, Any],
    started_at: str,
    finished_at: str,
    duration_seconds: float,
) -> Path:
    """Record core training evidence without running deployment benchmarks."""
    required_artifacts = (
        run_dir / "weights" / "best.pt",
        run_dir / "weights" / "last.pt",
        run_dir / "args.yaml",
        run_dir / "results.csv",
    )
    for required in required_artifacts:
        if not required.is_file():
            raise FileNotFoundError(f"Expected training artifact is missing: {required}")

    artifact_names = (
        "weights/best.pt",
        "weights/last.pt",
        "args.yaml",
        "results.csv",
        "results.png",
        "BoxPR_curve.png",
        "MaskPR_curve.png",
        "BoxF1_curve.png",
        "MaskF1_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
    )
    artifacts: dict[str, Any] = {}
    for relative_name in artifact_names:
        path = run_dir / relative_name
        if path.is_file():
            artifacts[relative_name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    record = {
        "scope": "software_v1.0_pytorch_model_training",
        "experiment_id": "E13",
        "status": "training_complete",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "project_root": str(ROOT),
        "run_directory": str(run_dir),
        "training_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "initial_weights": {
            "path": str(WEIGHTS),
            "bytes": WEIGHTS.stat().st_size,
            "sha256": sha256_file(WEIGHTS),
        },
        "data_yaml": {
            "path": str(DATA_YAML),
            "sha256": sha256_file(DATA_YAML),
        },
        "dataset_audit": dataset_audit,
        "training_arguments": config,
        "training_results": summarize_results_csv(
            run_dir / "results.csv", int(config["epochs"])
        ),
        "artifacts": artifacts,
        "deferred_after_v1.0": [
            "ONNX Runtime",
            "TensorRT FP32/FP16/INT8",
            "quantization and pruning",
            "cross-hardware benchmark",
            "RGB-D positioning and robot control",
        ],
    }
    path = run_dir / "training_run_record.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default=DEFAULT_RUN_NAME,
        help="output directory name under runs/segment",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="generate data.yaml, audit inputs, and print E13 parameters without training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from runs/segment/<name>/weights/last.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_data_yaml()
    dataset_audit = audit_dataset()

    config = e13_training_arguments(args.name)
    run_dir = Path(config["project"]) / str(config["name"])

    print(f"Pretrained weights: {WEIGHTS}")
    print(f"Dataset config: {DATA_YAML}")
    print(f"Run directory: {run_dir}")
    print(f"E13 configuration: {config}")

    if args.check_only:
        print("Check-only mode complete; training was not started.")
        return

    from ultralytics import YOLO

    if args.resume:
        checkpoint = run_dir / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint is missing: {checkpoint}")
        print(f"Resuming from: {checkpoint}")
        model = YOLO(str(checkpoint))
        train_kwargs = {"resume": True}
        record_config = {**config, "resume": True}
    else:
        if run_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing run: {run_dir}. "
                "Choose a new --name or use --resume."
            )
        model = YOLO(str(WEIGHTS))
        train_kwargs = config
        record_config = config

    started_at = now_iso()
    started = time.perf_counter()
    model.train(**train_kwargs)
    duration_seconds = time.perf_counter() - started
    finished_at = now_iso()

    record_path = write_training_record(
        run_dir,
        record_config,
        dataset_audit,
        started_at,
        finished_at,
        duration_seconds,
    )
    print(f"Training record saved: {record_path}")


if __name__ == "__main__":
    main()
