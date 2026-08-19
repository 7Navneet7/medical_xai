# 🏥 Medical XAI — Explainable Pneumonia Detection from Chest X-Rays

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI system that detects pneumonia from chest X-rays **and explains why** — using Grad-CAM, SHAP, and LIME together so predictions aren't a black box.

🔗 **Live demo:** [chest-disease-analysis.streamlit.app](https://chest-disease-analysis.streamlit.app/)

---

## Table of Contents
- [Why This Project](#why-this-project)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [XAI Methods](#xai-methods)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why This Project

Medical AI models can be accurate and still be untrustworthy, because clinicians have no way to see *why* a model reached a conclusion. This project addresses that "black box" problem directly: every diagnosis comes with three independent, cross-validating explanations, so a radiologist can check whether the model is actually looking at clinically relevant regions of the X-ray before trusting the output.

## Features

- **Pneumonia classification** from uploaded chest X-ray images using a fine-tuned DenseNet121
- **Three XAI methods, run together:**
  - **Grad-CAM** — visual attention heatmaps over the X-ray
  - **SHAP** — pixel-level, signed importance scores (evidence for/against pneumonia)
  - **LIME** — local, segment-based explanation of the specific prediction
- **Cross-method agreement view** — see where the three explanations do (or don't) agree, as a sanity check on the model
- **Confidence-aware UI** — color-coded confidence bands and probability distributions, not just a single label
- **Multi-framework backend** — TensorFlow/Keras and PyTorch models supported side by side
- **Streamlit interface** — drag-and-drop upload, tabbed views per XAI method, cached model loading for responsiveness

## Architecture

```
                    ┌─────────────────────────┐
                    │   Streamlit Frontend     │
                    │  (upload, tabs, charts)  │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   Preprocessing Layer     │
                    │  OpenCV + Albumentations  │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   DenseNet121 Model       │
                    │  (TensorFlow / PyTorch)   │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        ┌───────────┐     ┌───────────┐      ┌───────────┐
        │  Grad-CAM │     │   SHAP    │      │   LIME    │
        │  Heatmap  │     │ Features  │      │ Segments  │
        └───────────┘     └───────────┘      └───────────┘
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  Diagnosis + Explanation │
                    │        Dashboard          │
                    └─────────────────────────┘
```

## Tech Stack

| Category | Technologies |
|---|---|
| Frontend | Streamlit |
| Deep learning | TensorFlow / Keras, PyTorch |
| Computer vision | OpenCV, Albumentations, Pillow |
| Explainability | Grad-CAM, SHAP, LIME |
| Data / analysis | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit Cloud |

## Installation

```bash
# Clone the repo
git clone https://github.com/7Navneet7/medical_xai.git
cd medical_xai

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt        # Linux/Windows (CPU)
# or
pip install -r requirements_macos.txt  # macOS (Metal acceleration)

# Sanity check
python -c "import tensorflow as tf; import torch; import streamlit; print('OK')"
```

## Usage

```bash
streamlit run app.py
```

1. Upload a chest X-ray image (JPEG/PNG).
2. The model returns a diagnosis (Normal / Pneumonia) with a confidence score.
3. Switch between the Grad-CAM, SHAP, and LIME tabs to see how each method explains the prediction.
4. Check the agreement view to see how consistent the explanations are with each other.

## XAI Methods

**Grad-CAM** — backpropagates gradients of the predicted class into the last convolutional layer to produce a coarse heatmap of the regions the model attended to. Good for a fast visual sanity check.

**SHAP** — uses a gradient-based Shapley value approximation (`GradientExplainer`) to assign each pixel a signed contribution: positive (pushing toward pneumonia) or negative (pushing toward normal). Good for quantitative, per-pixel analysis.

**LIME** — perturbs superpixel segments of the image, observes how the prediction changes, and fits a local linear model to identify which segments mattered most for *this specific* prediction. Good for a locally faithful, model-agnostic explanation.

Running all three together lets you cross-check: if Grad-CAM, SHAP, and LIME all point at the same lung region, that's a much stronger signal than any single method alone.

## Project Structure

```
medical_xai/
├── app.py                   # Streamlit application entry point
├── config.py                # Configuration
├── requirements.txt         # Dependencies (Linux/Windows)
├── requirements_macos.txt   # Dependencies (macOS)
├── src/
│   ├── xai_engine.py        # Grad-CAM / SHAP / LIME implementations
│   ├── model_utils.py       # Model loading and inference
│   └── data_utils.py        # Preprocessing utilities
├── models/                  # Trained model weights
├── notebooks/                # Exploration / training notebooks
└── tests/                   # Unit tests
```

## Roadmap

- [ ] Additional XAI methods (Integrated Gradients, SmoothGrad)
- [ ] Natural-language explanation generation for clinicians
- [ ] DICOM support
- [ ] REST API deployment alongside the Streamlit app

## License

Released under the [MIT License](LICENSE).

---

Built as an exploration of explainable AI for medical imaging.
