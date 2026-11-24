from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    min_face_size: float = 0.03
    min_face_coverage: float = 0.06
    max_orientation_yaw: float = 35.0
    anti_spoof_threshold: float = 0.7
    quality_weight: float = 0.35
    comparison_threshold: float = 0.85
    detection_backend: str = "mtcnn"
    comparison_backend: str = "irse"
    use_real_time_mode: bool = True
