from __future__ import annotations

from pathlib import Path
from typing import Iterable


class RepresentationTrainer:
    """Training scaffold for synthetic spoof and representation-learning workflows."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir or "./artifacts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, dataset_root: str | Path) -> dict:
        dataset_path = Path(dataset_root)
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)

        manifest = {
            "dataset_root": str(dataset_path),
            "artifact_dir": str(self.output_dir),
            "status": "scaffolded",
            "notes": [
                "Use a synthetic spoof dataset such as SynthASpoof or a curated KYC spoof corpus.",
                "Train a lightweight embedding model with contrastive / representation-learning objectives.",
            ],
        }
        (self.output_dir / "training_manifest.json").write_text(str(manifest), encoding="utf-8")
        return manifest
