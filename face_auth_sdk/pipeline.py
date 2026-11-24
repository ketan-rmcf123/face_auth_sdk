from __future__ import annotations

from typing import Tuple

from .config import PipelineConfig
from .types import FaceAnalysis, FrameInput


class FaceLivenessPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def analyze_bbox(self, image_shape: Tuple[int, int], bbox: Tuple[int, int, int, int], face_count: int = 1) -> FaceAnalysis:
        return self.analyze_frame(FrameInput(image_shape=image_shape, bbox=bbox, face_count=face_count))

    def analyze_frame(self, frame: FrameInput) -> FaceAnalysis:
        height, width = frame.image_shape
        x, y, w, h = frame.bbox

        face_area = w * h
        frame_area = width * height
        coverage = face_area / float(frame_area)

        partial = (x < 0) or (y < 0) or (x + w > width) or (y + h > height) or (coverage < self.config.min_face_size)

        if w < 0.15 * width or h < 0.15 * height:
            orientation = "profile"
        else:
            orientation = "frontal"

        anti_spoof_score = 0.78 if orientation == "frontal" and not partial else 0.34
        quality_score = min(1.0, max(0.0, (coverage * 10.0) - (0.2 if partial else 0.0)))

        if anti_spoof_score >= self.config.anti_spoof_threshold and not partial and orientation == "frontal":
            verdict = "pass"
        else:
            verdict = "fail"

        return FaceAnalysis(
            face_count=frame.face_count,
            orientation=orientation,
            partial=partial,
            anti_spoof_score=anti_spoof_score,
            quality_score=quality_score,
            verdict=verdict,
        )
