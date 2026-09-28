# 🧠 NeuroScan.AI - Brain Tumor Detection & Segmentation

An advanced Deep Learning medical diagnostic platform and live interactive web application built with **Python**, **TensorFlow / tf-keras**, **OpenCV**, **Flask**, and **Vanilla JavaScript / CSS** for classifying and segmenting brain tumors from MRI scans.

---

## 🌟 Key Capabilities

- **Interactive Web Application & Live Demo**: Upload custom MRI scans or select from clinical test samples for instantaneous online diagnosis.
- **Dual-Stage Deep Learning Pipeline**:
  - **Stage 1 (Classification)**: Sequential CNN predicting tumor presence with 99.88% peak accuracy.
  - **Stage 2 (Spatial Localization)**: 19-layer U-Net encoder-decoder architecture generating millimeter-precise spatial tumor masks.
  - **Stage 3 (Morphometry)**: OpenCV contour tracing outlining perimeter boundaries and calculating quantitative tumor pixel surface area.
- **Multi-Modal Diagnostic Visualizations**:
  - **High-Contrast Contour Overlay**: Perimeter tracing with focal bounding boxes.
  - **Interactive Comparison Slider**: Drag-to-reveal before/after scan comparison.
  - **Thermal False-Color Heatmap**: JET colormap density gradient blended over the MRI scan.
  - **U-Net Binary Mask**: Raw neural network spatial probability localization.
- **Training Analytics Dashboard**: Interactive Chart.js graphs tracking accuracy and categorical loss convergence across epochs.
- **Desktop Tkinter Application**: Native Windows desktop interface included alongside the web platform.

---

## 📁 Repository Structure

```text
├── app.py                     # Flask web server & inference REST API
├── BrainTumor.py              # Native desktop Tkinter GUI application
├── run_web.bat                # One-click web application launcher
├── run.bat                    # One-click desktop GUI launcher
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html             # Web application user interface
├── static/
│   ├── css/style.css          # Modern dark clinical glassmorphic styling
│   └── js/main.js             # Client-side inference controller & Chart.js logic
├── Model/                     # Neural network checkpoints
│   ├── model.json             # CNN classification architecture
│   ├── model_weights.h5       # CNN classification trained weights
│   ├── segmented_model.json   # U-Net segmentation architecture
│   ├── segmented_weights.h5   # U-Net segmentation trained weights
│   ├── myimg_data.txt.npy     # Feature matrices
│   ├── myimg_label.txt.npy    # Target labels
│   └── history.pckl           # Training history checkpoint
├── brain_tumor_dataset/       # Training dataset ('yes' and 'no' classes)
└── testImages/                # Clinical test scans for demo inference
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10 - 3.12** installed on your system.

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/<YOUR_USERNAME>/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Application & Live Demo
Double-click **`run_web.bat`**, or run:
```bash
python app.py
```
Then open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

### 5. Launch the Desktop GUI (Alternative)
Double-click **`run.bat`**, or run:
```bash
python BrainTumor.py
```

---

## 📡 REST API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/status` | `GET` | Health check verifying model readiness |
| `/api/samples` | `GET` | Returns list of available sample MRI scans |
| `/api/sample/<filename>` | `GET` | Serves sample scan file |
| `/api/predict` | `POST` | Accepts image upload or sample name, runs dual-stage inference, and returns classifications, contour metrics, and base64 rendered overlays |
| `/api/metrics` | `GET` | Returns training epoch accuracy and loss history |

---

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow 2.x, tf-keras, Keras 3
- **Computer Vision**: OpenCV (`cv2`), Imutils
- **Backend**: Python Flask, Flask-CORS
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), Vanilla JavaScript (ES6+), Chart.js
