"""Run E13 instance segmentation on images, videos, or a live camera.

Examples:
    python detect_e13.py --source test/images/example.jpg
    python detect_e13.py --source test/images
    python detect_e13.py --source 1.mp4 --show
    python detect_e13.py --source 0

Press Q or Esc to stop video/camera inference when a preview window is open.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO


DEFAULT_WEIGHTS = (
    ROOT
    / "runs"
    / "segment"
    / "v18_e13_mosaic025"
    / "weights"
    / "best.pt"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "e13_detect"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv"}
WINDOW_NAME = "E13 tomato instance segmentation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="image, image directory, video, or camera index such as 0",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
        help="Ultralytics device, for example 0, 0,1, or cpu",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show a preview window; camera input is shown by default",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="disable the preview window, including for a camera",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="run inference without saving annotated images or video",
    )
    parser.add_argument("--camera-width", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="stop video/camera input after N frames; 0 means no limit",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.weights.expanduser().is_file():
        raise FileNotFoundError(f"Model weights not found: {args.weights}")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be between 0 and 1")
    if args.show and args.no_show:
        raise ValueError("--show and --no-show cannot be used together")
    if args.camera_width < 0 or args.camera_height < 0:
        raise ValueError("camera dimensions cannot be negative")
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")


def load_model(weights: Path) -> YOLO:
    model = YOLO(str(weights.expanduser().resolve()))
    # Use the precise business meaning in labels drawn on the output.
    names = dict(model.names)
    if 1 in names:
        names[1] = "main_peduncle"
    model.model.names = names
    return model


def predict_frame(
    model: YOLO,
    frame: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, dict[str, int]]:
    started = time.perf_counter()
    result = model.predict(
        source=frame,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        retina_masks=True,
        verbose=False,
    )[0]
    inference_ms = (time.perf_counter() - started) * 1000.0

    counts: dict[str, int] = {}
    if result.boxes is not None:
        for class_id in result.boxes.cls.int().cpu().tolist():
            class_name = str(result.names[class_id])
            counts[class_name] = counts.get(class_name, 0) + 1

    annotated = result.plot(
        boxes=True,
        masks=True,
        labels=True,
        conf=True,
        line_width=2,
    )
    add_status_bar(annotated, inference_ms, counts)
    return annotated, inference_ms, counts


def add_status_bar(
    image: np.ndarray,
    inference_ms: float,
    counts: dict[str, int],
) -> None:
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    count_text = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    text = f"{inference_ms:.1f} ms | {fps:.1f} FPS"
    if count_text:
        text += f" | {count_text}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, image.shape[1] / 1600.0)
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        image,
        (0, 0),
        (min(image.shape[1] - 1, text_width + 20), text_height + baseline + 16),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        image,
        text,
        (10, text_height + 7),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def read_image(path: Path) -> np.ndarray:
    """Read paths containing non-ASCII characters reliably on Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(path)


def image_files(source: Path) -> Iterable[Path]:
    if source.is_file():
        yield source
        return
    yield from sorted(
        path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def detect_images(
    model: YOLO,
    source: Path,
    args: argparse.Namespace,
) -> None:
    files = list(image_files(source))
    if not files:
        raise FileNotFoundError(f"No supported images found: {source}")

    output_root = args.output_dir.expanduser().resolve() / "images"
    show = args.show and not args.no_show
    total_ms = 0.0
    processed = 0

    try:
        for index, image_path in enumerate(files, start=1):
            frame = read_image(image_path)
            annotated, inference_ms, counts = predict_frame(model, frame, args)
            total_ms += inference_ms
            processed += 1

            if not args.no_save:
                relative = image_path.name if source.is_file() else image_path.relative_to(source)
                target = output_root / Path(relative).with_name(
                    f"{Path(relative).stem}_e13{Path(relative).suffix}"
                )
                write_image(target, annotated)
                saved_text = f" -> {target}"
            else:
                saved_text = ""

            print(
                f"[{index}/{len(files)}] {image_path.name}: "
                f"{inference_ms:.1f} ms, detections={sum(counts.values())}{saved_text}"
            )

            if show:
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(0) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
    finally:
        if show:
            cv2.destroyAllWindows()

    print(f"Processed {processed} image(s); average inference {total_ms / processed:.1f} ms")


def open_capture(source: str, is_camera: bool) -> cv2.VideoCapture:
    if is_camera:
        camera_index = int(source)
        if os.name == "nt":
            capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(camera_index)
        else:
            capture = cv2.VideoCapture(camera_index)
    else:
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        kind = "camera" if is_camera else "video"
        raise RuntimeError(f"Could not open {kind}: {source}")
    return capture


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    return writer


def detect_video_or_camera(
    model: YOLO,
    source: str,
    is_camera: bool,
    args: argparse.Namespace,
) -> None:
    capture = open_capture(source, is_camera)
    if is_camera:
        if args.camera_width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        if args.camera_height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_camera else 0
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Input reports an invalid frame size")
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    show = (is_camera or args.show) and not args.no_show
    writer: cv2.VideoWriter | None = None
    output_path: Path | None = None
    if not args.no_save:
        filename = (
            f"camera_{source}_e13.mp4"
            if is_camera
            else f"{Path(source).stem}_e13.mp4"
        )
        output_path = args.output_dir.expanduser().resolve() / "videos" / filename
        writer = open_writer(output_path, fps, (width, height))

    frame_index = 0
    total_inference_ms = 0.0
    started = time.perf_counter()
    print("Press Q or Esc in the preview window to stop.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            annotated, inference_ms, counts = predict_frame(model, frame, args)
            total_inference_ms += inference_ms
            frame_index += 1

            if writer is not None:
                writer.write(annotated)
            if show:
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break

            if frame_index == 1 or frame_index % 30 == 0:
                progress = f"/{total_frames}" if total_frames > 0 else ""
                print(
                    f"Frame {frame_index}{progress}: {inference_ms:.1f} ms, "
                    f"detections={sum(counts.values())}"
                )
            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if frame_index == 0:
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise RuntimeError("No frames were read from the input")

    elapsed = time.perf_counter() - started
    print(f"Processed {frame_index} frame(s) in {elapsed:.1f} seconds")
    print(f"Average model inference: {total_inference_ms / frame_index:.1f} ms")
    if output_path is not None:
        print(f"Saved result: {output_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    model = load_model(args.weights)

    source_text = args.source.strip()
    is_camera = source_text.isdigit()
    if is_camera:
        detect_video_or_camera(model, source_text, True, args)
        return

    source_path = Path(source_text).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")
    if source_path.is_dir() or source_path.suffix.lower() in IMAGE_SUFFIXES:
        detect_images(model, source_path, args)
    elif source_path.is_file() and source_path.suffix.lower() in VIDEO_SUFFIXES:
        detect_video_or_camera(model, str(source_path), False, args)
    else:
        raise ValueError(f"Unsupported source type: {source_path}")


if __name__ == "__main__":
    main()
