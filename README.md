---
title: VigiLens
emoji: 🚦
colorFrom: blue
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
---

# VigiLens: Real-Time Traffic Violation Vision Pipeline

![VigiLens Architecture](https://img.shields.io/badge/Architecture-Modular-blue.svg) ![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg) ![YOLO11](https://img.shields.io/badge/YOLO-11-yellow.svg)

**VigiLens** is an end-to-end computer vision pipeline designed for real-time traffic violation detection and automated enforcement evidence generation. 

Automated traffic systems frequently fail in the real world due to bad weather, low lighting, and the immense computational cost of running multiple deep-learning models simultaneously. VigiLens was architected to solve these exact bottlenecks using a mix of zero-shot preprocessing, high-speed spatial math heuristics, and an advanced OCR cascade.

## 🚀 Key Features

*   **Phase 1: Smart Image Preprocessing Router:** In less than 5 milliseconds, the system evaluates frames for brightness, blur, and contrast. It dynamically routes degraded frames through Zero-DCE (a zero-reference low-light enhancement PyTorch model) or CLAHE before strictly letterboxing them to preserve aspect ratios.
*   **Phase 2: Object Detection (YOLO11):** Utilizes the YOLO11 Large model with its C2PSA module, which excels at detecting highly occluded vehicles and pedestrians in dense traffic.
*   **Phase 3: Modular Violation Rules Engine:** Instead of running a separate neural network for every violation, VigiLens infers behaviors using lightning-fast spatial mathematics. It enforces:
    *   **Overcrowding / Triple Riding**
    *   **Stop-Line & Red-Light Crossing**
    *   **Hazardous Loads**
    *   **Helmet & Seatbelt Non-Compliance**
    *   **Wrong-Side Driving & Speeding**
*   **Phase 4: Hybrid OCR Cascades:** Generates actionable evidence by strictly cropping the offending vehicle and passing it through a cascading OCR system (Gemini Vision / Google Cloud Vision $\rightarrow$ EasyOCR $\rightarrow$ PaddleOCR) with strict RTO Regex validation.
*   **Streamlit Backend:** Non-technical officers can draw virtual constraint polygons interactively, set Area Profiles, and review flagged evidence in a Human-In-The-Loop queue.

---

## 📁 Repository Structure

```text
├── app.py                 # Main Streamlit frontend and UI orchestrator
├── app_data/              # Contains the SQLite database for the evidence review queue
├── models/                # Local model weights (YOLO11, EasyOCR)
├── output_evidence/       # Directory where cropped violation evidence is saved
├── output_samples/        # Processed sample outputs demonstrating the pipeline
├── src/                   # Core source code containing the 4-phase modules and rules engine
├── test_images/           # ⚠️ SAMPLE IMAGES: Use these images to test the pipeline!
├── detailed_report.pdf    # Comprehensive 5-page academic/technical report
└── requirements.txt       # Python dependencies required for deployment
```

---

## ⚙️ Installation & Setup

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda to isolate the project environment.

### 1. Create a Conda Environment
Open your terminal and create a fresh Python 3.10 environment:
```bash
conda create -n vigilens python=3.10 -y
conda activate vigilens
```

### 2. Install PyTorch
Install PyTorch with CUDA support (recommended for GPU acceleration) or CPU. Refer to the [PyTorch website](https://pytorch.org/get-started/locally/) for your specific system constraints.
```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Dependencies
The system relies on Ultralytics (YOLO), OpenCV, Streamlit, and multiple OCR engines. Install them via pip:
```bash
pip install ultralytics opencv-python streamlit
pip install easyocr paddleocr paddlepaddle
pip install google-cloud-vision google-genai
pip install numpy pandas sqlite3
```
*(Note: If you plan on using the Gemini or Google Cloud Vision OCR fallbacks, ensure your API keys are configured in your environment variables).*

---

## 🖥️ Usage Guide

Once your environment is activated and dependencies are installed, you can launch the unified backend and UI.

### Step 1: Export Model Path & Start the Streamlit Application
Before launching the application, you must set the environment variable pointing to your downloaded YOLO model. Run this in the same terminal session so the pipeline can identify it:
```bash
export GRIDLOCK_PLATE_MODEL_PATH=./models/yolo11l.pt
streamlit run app.py
```

### Step 2: Configure Area Profiles
1. Open the local web address provided by Streamlit (usually `http://localhost:8501`).
2. Navigate to the **Configuration** tab.
3. Draw virtual polygons over your camera feed to define crosswalks, stop-lines, and no-parking zones.
4. Set your contextual Area Profiles (e.g., Highway vs. School Zone limits).

### Step 3: Run the Pipeline
1. Upload a video feed or connect your RTSP stream.
2. The `main_pipeline.py` will automatically route frames through Phase 1-4.
3. Switch to the **Human Review** tab to view flagged violations. Here, you can examine the cropped evidence, the OCR read, and the bounding box logic before choosing to **Approve** or **Reject** the ticket.

---

## 🏗️ Architecture Extensibility

VigiLens uses a strict Plugin Architecture for its Rule Engine. To add a new traffic rule:
1. Create a new class extending `BaseViolationRule`.
2. Drop the `.py` file into `src/rules/plugins/`.
3. The engine will dynamically register and apply your spatial mathematics to the YOLO bounding boxes without needing core pipeline modifications.
