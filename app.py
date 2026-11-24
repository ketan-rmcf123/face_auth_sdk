import argparse
from pathlib import Path

from face_auth_sdk import FaceComparator, FaceDetector, FaceLivenessPipeline, PipelineConfig, RepresentationTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configurable face-auth liveness workflow")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--x", type=int, default=330)
    parser.add_argument("--y", type=int, default=180)
    parser.add_argument("--w", type=int, default=220)
    parser.add_argument("--h", type=int, default=220)
    parser.add_argument("--min-face-size", type=float, default=0.03)
    parser.add_argument("--anti-spoof-threshold", type=float, default=0.7)
    parser.add_argument("--train", type=str, default="", help="Optional dataset root to scaffold a training manifest")
    parser.add_argument("--image", type=str, default="", help="Optional image path for detector/comparison demo")
    args = parser.parse_args()

    config = PipelineConfig(min_face_size=args.min_face_size, anti_spoof_threshold=args.anti_spoof_threshold)
    pipeline = FaceLivenessPipeline(config)
    analysis = pipeline.analyze_bbox((args.height, args.width), (args.x, args.y, args.w, args.h))

    print("Face liveness analysis")
    print(f"Face count: {analysis.face_count}")
    print(f"Orientation: {analysis.orientation}")
    print(f"Partial: {analysis.partial}")
    print(f"Anti-spoof score: {analysis.anti_spoof_score:.2f}")
    print(f"Quality score: {analysis.quality_score:.2f}")
    print(f"Verdict: {analysis.verdict}")

    if args.image:
        detector = FaceDetector()
        comparator = FaceComparator()
        faces = detector.detect_faces(Path(args.image))
        print(f"Detected faces via {detector.backend}: {len(faces)}")
        if faces:
            embedding = comparator.generate_embedding(args.image)
            print(f"Embedding shape: {embedding.shape}")

    if args.train:
        trainer = RepresentationTrainer(output_dir="./artifacts")
        manifest = trainer.train(args.train)
        print(f"Training scaffold created: {manifest['artifact_dir']}")


if __name__ == "__main__":
    main()
