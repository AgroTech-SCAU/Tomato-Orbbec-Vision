"""2D detector wrapper used by the unified vision node."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    """One 2D detection in image coordinates."""

    semantic_label: str
    class_id: int
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


class YoloDetector:
    """Thin YOLO wrapper so ROS transport is kept out of detector logic."""

    def __init__(self, model_path: str, device: str):
        self._device = device
        self._model = YOLO(model_path)

    def predict(
        self,
        image_bgr: np.ndarray,
        confidence_floor: float,
        class_filter: Iterable[str] = (),
    ) -> list[Detection]:
        wanted = {name.strip().lower() for name in class_filter if name.strip()}
        result = self._model.predict(
            source=image_bgr,
            conf=float(confidence_floor),
            device=self._device,
            verbose=False,
        )[0]

        if result.boxes is None:
            return []

        height, width = image_bgr.shape[:2]
        xyxy = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        names = result.names

        detections: list[Detection] = []
        for box, class_id, score in zip(xyxy, classes, scores):
            label = str(names[int(class_id)])
            if wanted and label.lower() not in wanted:
                continue

            xmin, ymin, xmax, ymax = [int(round(value)) for value in box]
            xmin = max(0, min(width - 1, xmin))
            xmax = max(0, min(width - 1, xmax))
            ymin = max(0, min(height - 1, ymin))
            ymax = max(0, min(height - 1, ymax))
            if xmax < xmin or ymax < ymin:
                continue

            detections.append(
                Detection(
                    semantic_label=label,
                    class_id=int(class_id),
                    confidence=float(score),
                    bbox_xyxy=(xmin, ymin, xmax, ymax),
                )
            )
        return detections
