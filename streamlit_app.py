"""
FaceAuth SDK: Interactive Face Liveness, Quality & 1:1 Verification Studio.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from face_auth_sdk import FaceComparator, FaceDetector, FaceLivenessPipeline, PipelineConfig

st.set_page_config(
    page_title="FaceAuth Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ FaceAuth SDK: Liveness & Verification Studio")
st.caption("Biometric facial quality analysis, anti-spoofing liveness verification, and 1:1 facial recognition.")

# Sidebar parameters
with st.sidebar:
    st.header("⚙️ Liveness Thresholds")
    min_face_size = st.slider("Minimum Face Coverage (%)", 1.0, 20.0, 3.0, 0.5) / 100.0
    anti_spoof_thresh = st.slider("Anti-Spoof Threshold", 0.0, 1.0, 0.70, 0.05)
    match_threshold = st.slider("1:1 Verification Match Threshold", 0.50, 0.99, 0.85, 0.01)

config = PipelineConfig(min_face_size=min_face_size, anti_spoof_threshold=anti_spoof_thresh)
pipeline = FaceLivenessPipeline(config)
detector = FaceDetector()
comparator = FaceComparator()

tabs = st.tabs([
    "👤 Live Liveness & Anti-Spoofing",
    "🔐 1:1 Face Verification",
    "📊 Pipeline Inspector & Test Runner",
    "📖 API Documentation"
])

# ----------------- TAB 1: LIVENESS -----------------
with tabs[0]:
    st.header("Facial Liveness & Quality Assessment")
    st.write("Capture or upload a face to evaluate bounding box orientation, coverage, anti-spoofing score, and pass/fail verdict.")

    photo = st.camera_input("Capture Camera Snapshot")
    uploaded = st.file_uploader("Or Upload an Image", type=["jpg", "jpeg", "png", "webp"], key="liveness_upload")

    eval_image = None
    if photo:
        eval_image = Image.open(photo).convert("RGB")
    elif uploaded:
        eval_image = Image.open(uploaded).convert("RGB")
    else:
        if st.button("🧪 Generate Synthetic Frontal Face"):
            # Create synthetic portrait
            test_img = Image.new("RGB", (640, 480), color=(240, 240, 245))
            draw = ImageDraw.Draw(test_img)
            # Head oval
            draw.ellipse([240, 120, 400, 320], fill=(220, 185, 155), outline=(180, 140, 110), width=3)
            # Eyes
            draw.ellipse([275, 180, 295, 200], fill=(50, 50, 50))
            draw.ellipse([345, 180, 365, 200], fill=(50, 50, 50))
            # Mouth
            draw.arc([290, 240, 350, 270], start=0, end=180, fill=(160, 60, 60), width=4)
            eval_image = test_img

    if eval_image is not None:
        w, h = eval_image.size
        # Detect faces
        faces = detector.detect_faces(eval_image)
        st.write(f"Detected **{len(faces)}** face candidate(s).")

        # Pick primary face
        primary_box = faces[0]["bbox"] if faces else (w // 4, h // 4, w // 2, h // 2)
        x, y, bw, bh = primary_box

        analysis = pipeline.analyze_bbox((h, w), (x, y, bw, bh), face_count=len(faces))

        # Draw HUD annotation
        annotated = eval_image.copy()
        draw = ImageDraw.Draw(annotated)
        box_color = (40, 200, 60) if analysis.verdict == "pass" else (240, 60, 60)
        draw.rectangle([x, y, x + bw, y + bh], outline=box_color, width=4)
        draw.text((x + 8, max(0, y - 24)), f"{analysis.verdict.upper()} | Anti-Spoof: {analysis.anti_spoof_score:.2f}", fill=box_color)

        col_img, col_metrics = st.columns([3, 2])
        with col_img:
            st.image(annotated, caption="Liveness Telemetry HUD", use_container_width=True)

        with col_metrics:
            st.subheader("Biometric Assessment")
            if analysis.verdict == "pass":
                st.success("✅ **LIVENESS VERDICT: PASS**")
            else:
                st.error("❌ **LIVENESS VERDICT: FAIL**")

            st.metric("Anti-Spoof Score", f"{analysis.anti_spoof_score:.2f}")
            st.metric("Quality Score", f"{analysis.quality_score:.2f}")
            st.metric("Orientation", analysis.orientation.upper())
            st.metric("Partial Face Detected", "YES (Alert)" if analysis.partial else "NO (Full Face)")

# ----------------- TAB 2: VERIFICATION -----------------
with tabs[1]:
    st.header("1:1 Face Verification & Matching Studio")
    st.write("Compare two faces (Reference and Probe) to calculate biometric similarity and verify identity.")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("Reference Identity (ID / Document)")
        file_ref = st.file_uploader("Upload Reference Face", type=["jpg", "png", "webp"], key="ref_face")
        if file_ref:
            img_ref = Image.open(file_ref).convert("RGB")
            st.image(img_ref, width=220)
        else:
            img_ref = Image.new("RGB", (128, 128), color=(180, 140, 120))
            st.image(img_ref, caption="Sample Template A", width=220)

    with col_v2:
        st.subheader("Probe Image (Live Selfie)")
        file_probe = st.file_uploader("Upload Live Probe Face", type=["jpg", "png", "webp"], key="probe_face")
        if file_probe:
            img_probe = Image.open(file_probe).convert("RGB")
            st.image(img_probe, width=220)
        else:
            img_probe = img_ref.copy()
            st.image(img_probe, caption="Sample Template B (Same)", width=220)

    if st.button("🔍 Verify Match", type="primary"):
        emb_ref = comparator.generate_embedding(img_ref)
        emb_probe = comparator.generate_embedding(img_probe)
        similarity = comparator.compare_embeddings(emb_ref, emb_probe)

        st.markdown("### Verification Result")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("Cosine Similarity", f"{similarity * 100.0:.1f}%")
            st.progress(similarity)

        with col_r2:
            if similarity >= match_threshold:
                st.success(f"🎉 **MATCH CONFIRMED** (Similarity ≥ {int(match_threshold*100)}%)")
                st.caption("Biometric identity verified with high confidence.")
            else:
                st.error(f"⛔ **NO MATCH** (Similarity < {int(match_threshold*100)}%)")
                st.caption("Biometric features differ beyond allowed threshold.")

# ----------------- TAB 3: INSPECTOR -----------------
with tabs[2]:
    st.header("SDK Pipeline Configuration Inspector")
    st.json({
        "min_face_size": config.min_face_size,
        "anti_spoof_threshold": config.anti_spoof_threshold,
        "detector_backend": detector.backend,
        "comparator_backend": comparator.backend,
        "embedding_dimensions": 64 * 64 * 3
    })

# ----------------- TAB 4: API DOCS -----------------
with tabs[3]:
    st.header("FaceAuth Python SDK API Reference")
    st.markdown("""
    ```python
    from face_auth_sdk import FaceLivenessPipeline, FaceComparator, PipelineConfig

    # 1. Initialize pipeline
    pipeline = FaceLivenessPipeline(PipelineConfig(min_face_size=0.03, anti_spoof_threshold=0.7))

    # 2. Analyze liveness
    analysis = pipeline.analyze_bbox(image_shape=(1080, 720), bbox=(330, 180, 220, 220))
    print(analysis.verdict)  # "pass" or "fail"

    # 3. 1:1 Biometric Verification
    comparator = FaceComparator()
    emb_a = comparator.generate_embedding("id_photo.jpg")
    emb_b = comparator.generate_embedding("selfie.jpg")
    similarity = comparator.compare_embeddings(emb_a, emb_b)
    print(f"Similarity: {similarity * 100:.1f}%")
    ```
    """)
