# X-Brain: Explainable AI Brain Tumor Analysis System

## Project Structure
```
xbrain/
├── models/
│   ├── classifier.py        ← EfficientNet-B0 + Grad-CAM
│   └── segmentor.py         ← SwinUNETR segmentation
├── utils/
│   ├── image_utils.py       ← Image encode/decode helpers
│   └── clinical_knowledge.py← Medical knowledge base + report generator
├── api/
│   └── main.py              ← FastAPI backend (POST /analyze)
├── frontend/
│   └── app.py               ← Streamlit frontend
├── checkpoints/             ← PUT YOUR MODEL WEIGHTS HERE
│   ├── EfficientNetB0_BrainTumor_full.weights.h5
│   └── SwinUNETR_Segmentation_best.pth
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your trained model weights
```
checkpoints/
├── EfficientNetB0_BrainTumor_full.weights.h5   ← from classification.ipynb
└── SwinUNETR_Segmentation_best.pth             ← from segmentation.ipynb
```

### 3. Run FastAPI backend (Terminal 1)
```bash
cd xbrain
uvicorn api.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

### 4. Run Streamlit frontend (Terminal 2)
```bash
cd xbrain
streamlit run frontend/app.py
```
App: http://localhost:8501

## Pipeline Flow
Upload MRI → EfficientNet-B0 Classification → Grad-CAM (XAI) → SwinUNETR Segmentation (skipped if no tumor) → Tumor Area Stats → Clinical Report

## Disclaimer
For research/educational use only. Not a clinical diagnostic tool.
