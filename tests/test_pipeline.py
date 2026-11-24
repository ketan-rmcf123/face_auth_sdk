import os
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face_auth_sdk import FaceComparator, FaceDetector, FaceLivenessPipeline, PipelineConfig


class PipelineTests(unittest.TestCase):
    def test_frontal_face_box_is_not_partial_and_orientation_frontal(self):
        pipeline = FaceLivenessPipeline()
        analysis = pipeline.analyze_bbox((1080, 720), (330, 180, 220, 220))

        self.assertEqual(analysis.orientation, "frontal")
        self.assertFalse(analysis.partial)
        self.assertGreaterEqual(analysis.anti_spoof_score, 0.0)

    def test_edge_face_box_is_partial_and_profile(self):
        pipeline = FaceLivenessPipeline()
        analysis = pipeline.analyze_bbox((1080, 720), (10, 140, 90, 140))

        self.assertEqual(analysis.orientation, "profile")
        self.assertTrue(analysis.partial)
        self.assertLess(analysis.anti_spoof_score, 0.5)

    def test_configurable_thresholds_can_fail_a_box(self):
        pipeline = FaceLivenessPipeline(PipelineConfig(min_face_size=0.5))
        analysis = pipeline.analyze_bbox((1080, 720), (330, 180, 220, 220))

        self.assertEqual(analysis.verdict, "fail")

    def test_comparer_returns_high_similarity_for_same_face(self):
        comparator = FaceComparator()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "face.png"
            Image.new("RGB", (64, 64), color=(120, 80, 70)).save(path)
            embedding_a = comparator.generate_embedding(path)
            embedding_b = comparator.generate_embedding(path)
            similarity = comparator.compare_embeddings(embedding_a, embedding_b)

        self.assertGreaterEqual(similarity, 0.95)

    def test_comparer_different_images(self):
        comparator = FaceComparator()
        img_a = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img_b = Image.new("RGB", (80, 80), color=(0, 255, 0))
        emb_a = comparator.generate_embedding(img_a)
        emb_b = comparator.generate_embedding(img_b)
        sim = comparator.compare_embeddings(emb_a, emb_b)
        self.assertLess(sim, 0.5)

    def test_detector_fallback(self):
        detector = FaceDetector()
        img = Image.new("RGB", (200, 200), color=(100, 100, 100))
        faces = detector.detect_faces(img)
        self.assertGreater(len(faces), 0)
        self.assertIn("bbox", faces[0])


if __name__ == "__main__":
    unittest.main()
