from __future__ import annotations

from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image


class FaceComparator:
    """A lightweight IRSE-style embedding comparator scaffold with auto-resizing."""

    def __init__(self, backend: str = "irse", target_size: tuple[int, int] = (64, 64)) -> None:
        self.backend = backend
        self.target_size = target_size

    def generate_embedding(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(path)
            img = Image.open(path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Normalize resolution so embeddings have identical dimensions
        resized = img.resize(self.target_size, Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0

        # L2-normalized 1D feature vector
        flat = arr.reshape(-1)
        norm = np.linalg.norm(flat)
        if norm > 1e-6:
            flat = flat / norm
        return flat

    def compare_embeddings(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        if embedding_a.shape != embedding_b.shape:
            raise ValueError(f"Embeddings must have the same shape, got {embedding_a.shape} and {embedding_b.shape}")
        similarity = float(np.dot(embedding_a, embedding_b))
        return max(0.0, min(1.0, similarity))
