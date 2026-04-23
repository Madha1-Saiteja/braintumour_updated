# BrainTumour_Updated

A full-stack brain tumor analysis project with a FastAPI backend and a Vite React frontend.

## Project Structure

- `Backend/` — Python backend for model inference, RAG pipeline, and API endpoints.
  - `xbrain/` — main FastAPI application and model code.
  - `xbrain/api/main.py` — FastAPI app entry point.
  - `xbrain/models/` — classifier and segmentor model implementations.
  - `xbrain/checkpoints/` — pretrained model weights and FAISS index.
- `Frontend/` — React + Vite TypeScript web app.
  - `src/` — main application code and UI components.
  - `public/` — static assets.

## What this project does

- Uploads a brain MRI image
- Runs tumor classification with EfficientNet-B0
- Generates Grad-CAM explainability visualizations
- Builds a clinical report and RAG-enhanced explanation
- Displays metrics such as confidence, inference speed, and tumor coverage

## Backend Setup

### 1. Create the Python environment

```powershell
cd C:\Users\navay\Downloads\braintumour_updated\Backend\xbrain
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment

Copy `Backend/xbrain/.env` to `Backend/xbrain/.env` if needed and verify values.

Important variables:

- `GROQ_API_KEY` — Groq API key for RAG reports
- `API_HOST` — backend host (usually `localhost`)
- `API_PORT` — backend port (usually `8000`)
- `CLF_WEIGHTS` — classifier weights path
- `SEG_WEIGHTS` — segmentor weights path
- `DOCS_DIR` — medical docs folder for RAG indexing
- `FAISS_INDEX_PATH` — FAISS vector store path

### 4. Run backend

```powershell
cd C:\Users\navay\Downloads\braintumour_updated\Backend\xbrain
.\.venv\Scripts\Activate.ps1
python -m uvicorn --app-dir xbrain api.main:app --reload --port 8000
```

If the backend starts successfully, it should be available at:

```
http://127.0.0.1:8000
```

## Frontend Setup

### 1. Install dependencies

```powershell
cd C:\Users\navay\Downloads\braintumour_updated\Frontend
npm install
```

### 2. Run frontend

```powershell
npm run dev
```

The frontend should then be available at:

```
http://localhost:8080
```

## Usage

1. Start the backend first.
2. Start the frontend.
3. Open the frontend in your browser.
4. Upload a brain MRI image.
5. View classification, Grad-CAM, metrics, and explanations.

## Notes

- If the segmentation model checkpoint is missing or invalid, the app may still run with segmentation disabled.
- Do not commit `.env` files or private API keys.
- The root `.gitignore` already excludes virtual environments, environment files, node modules, and model checkpoints.

## GitHub repository

If you push this project to GitHub, use the repository URL:

```
https://github.com/Madha1-Saiteja/braintumour_updated.git
```

## Contact

For questions or fixes, open an issue in your GitHub repository or update the code/configuration directly.
