from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FaceAnalysis:
    face_count: int
    orientation: str
    partial: bool
    anti_spoof_score: float
    quality_score: float
    verdict: str


@dataclass
class FrameInput:
    image_shape: tuple[int, int]
    bbox: tuple[int, int, int, int]
    face_count: int = 1
