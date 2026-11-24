# 🛡️ FaceAuth SDK & Biometric Liveness Studio

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Interface-Streamlit-red.svg)](https://streamlit.io)
[![Tests: Passing](https://img.shields.io/badge/Tests-6%20Passed-brightgreen.svg)]()

> A robust, modular facial authentication and anti-spoofing liveness SDK supporting **bounding box quality scoring**, **orientation tracking**, **anti-spoof detection**, and **1:1 facial verification**.

---

## 🌟 Key Features

- **👁️ Face Liveness & Quality Pipeline**:
  - Orientation classification (frontal vs profile).
  - Partial / edge occlusion detection.
  - Coverage ratio calculation against configurable `min_face_size`.
  - Anti-spoof confidence scoring with binary Pass/Fail verdict.
- **🔐 1:1 Biometric Face Verification**:
  - Normalized feature embedding generation.
  - Cosine similarity matching between reference IDs and live probe selfies.
  - Configurable verification threshold.
- **💻 Interactive Streamlit Studio**:
  - Live webcam portrait testing and photo upload.
  - Real-time HUD bounding box annotations.
  - Interactive 1:1 face matcher with similarity progress bar.
- **⚡ Developer-Friendly SDK**:
  - Fully typed with docstrings and zero mandatory external C dependencies.

---

## 🚀 Quick Start

### 1. Installation

```bash
cd face_auth_sdk
pip install -r requirements.txt
```

### 2. Launch Interactive Web Studio

```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501`.

### 3. Command Line Interface (CLI)

```bash
python app.py --width 1080 --height 720 --x 330 --y 180 --w 220 --h 220
```

---

## 🧪 Testing

Run the automated test suite:

```bash
python -m unittest discover -s tests
```

All 6 unit tests validate frontal vs profile orientation, partial box detection, configurable thresholds, and 1:1 similarity matching.

---

## 📜 License

MIT License. Open for biometric and security research.
