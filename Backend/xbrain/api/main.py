"""
api/main.py
FastAPI backend for X-Brain.

This version keeps startup lightweight for low-memory hosts such as
Render free instances. Heavy ML modules are imported and models are loaded
only when an endpoint actually needs them.
"""

import importlib
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.clinical_knowledge import generate_clinical_report
from utils.image_utils import mask_to_base64, ndarray_to_base64, read_image_from_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("xbrain")

CLF_WEIGHTS = ROOT / os.getenv(
    "CLF_WEIGHTS", "checkpoints/EfficientNetB0_BrainTumor_full.weights.h5"
)
SEG_WEIGHTS = ROOT / os.getenv(
    "SEG_WEIGHTS", "checkpoints/SwinUNETR_Segmentation_best.pth"
)
FAISS_INDEX_PATH = ROOT / os.getenv("FAISS_INDEX_PATH", "checkpoints/faiss_index.index")

MODELS = {}
_classifier_module = None
_segmentor_module = None
_rag_module = None


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


ENABLE_CLASSIFIER = _env_flag("ENABLE_CLASSIFIER", True)
ENABLE_SEGMENTOR = _env_flag("ENABLE_SEGMENTOR", False)
ENABLE_RAG = _env_flag("ENABLE_RAG", False)
EAGER_LOAD_MODELS = _env_flag("EAGER_LOAD_MODELS", False)
BUILD_RAG_INDEX_ON_STARTUP = _env_flag("BUILD_RAG_INDEX_ON_STARTUP", False)


def _resolve_existing_path(configured_path: Path, pattern: str) -> Path:
    if configured_path.exists():
        return configured_path

    search_dirs = [
        ROOT / "checkpoints",
        ROOT.parent / "xbrain" / "checkpoints",
        ROOT.parent / "Backend" / "xbrain" / "checkpoints",
        Path.cwd() / "checkpoints",
        Path.cwd() / "xbrain" / "checkpoints",
        Path.cwd() / "Backend" / "xbrain" / "checkpoints",
    ]

    seen = set()
    for directory in search_dirs:
        directory_key = str(directory.resolve()) if directory.exists() else str(directory)
        if directory_key in seen:
            continue
        seen.add(directory_key)

        if not directory.exists():
            continue

        matches = sorted(directory.glob(pattern))
        if matches:
            resolved = matches[0]
            log.warning(
                "Configured artifact path missing: %s. Using fallback: %s",
                configured_path,
                resolved,
            )
            return resolved

    return configured_path


def _get_classifier_module():
    global _classifier_module
    if _classifier_module is None:
        _classifier_module = importlib.import_module("models.classifier")
    return _classifier_module


def _get_segmentor_module():
    global _segmentor_module
    if _segmentor_module is None:
        _segmentor_module = importlib.import_module("models.segmentor")
    return _segmentor_module


def _get_rag_module():
    global _rag_module
    if _rag_module is None:
        _rag_module = importlib.import_module("utils.rag_pipeline")
    return _rag_module


def _rag_index_exists() -> bool:
    return FAISS_INDEX_PATH.exists()


def _ensure_classifier_loaded():
    weights_path = _resolve_existing_path(CLF_WEIGHTS, "*.weights.h5")

    if not ENABLE_CLASSIFIER:
        raise HTTPException(status_code=503, detail="Classifier is disabled by configuration.")

    if "classifier" in MODELS:
        return MODELS["classifier"]

    if not weights_path.exists():
        raise HTTPException(
            status_code=503, detail=f"Classification weights not found: {weights_path}"
        )

    log.info("Loading classifier on demand")
    classifier_module = _get_classifier_module()
    MODELS["classifier"] = classifier_module.build_classifier(str(weights_path))
    log.info("Classifier loaded")
    return MODELS["classifier"]


def _ensure_segmentor_loaded():
    weights_path = _resolve_existing_path(SEG_WEIGHTS, "*.pth")

    if not ENABLE_SEGMENTOR:
        return None, None

    if "segmentor" in MODELS and "device" in MODELS:
        return MODELS["segmentor"], MODELS["device"]

    if not weights_path.exists():
        log.warning("Segmentation weights not found: %s", weights_path)
        return None, None

    log.info("Loading segmentor on demand")
    segmentor_module = _get_segmentor_module()
    model_seg, device = segmentor_module.build_segmentor(str(weights_path))
    MODELS["segmentor"] = model_seg
    MODELS["device"] = device
    log.info("Segmentor loaded")
    return model_seg, device


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting X-Brain API")
    log.info(
        "Startup config: eager_load=%s classifier=%s segmentor=%s rag=%s",
        EAGER_LOAD_MODELS,
        ENABLE_CLASSIFIER,
        ENABLE_SEGMENTOR,
        ENABLE_RAG,
    )

    if EAGER_LOAD_MODELS and ENABLE_CLASSIFIER:
        try:
            _ensure_classifier_loaded()
        except Exception as exc:
            log.warning("Classifier preload skipped: %s", exc)

    if EAGER_LOAD_MODELS and ENABLE_SEGMENTOR:
        try:
            _ensure_segmentor_loaded()
        except Exception as exc:
            log.warning("Segmentor preload skipped: %s", exc)

    if BUILD_RAG_INDEX_ON_STARTUP and ENABLE_RAG:
        try:
            _get_rag_module().build_index(force=False)
        except Exception as exc:
            log.warning("Index build skipped: %s", exc)

    log.info("X-Brain API ready to accept traffic")
    yield
    MODELS.clear()
    log.info("Models unloaded")


app = FastAPI(
    title="X-Brain API",
    description="Explainable AI Brain Tumor Analysis API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClassificationResult(BaseModel):
    class_name: str
    class_idx: int
    confidence: float
    probabilities: dict
    has_tumor: bool


class SegmentationResult(BaseModel):
    tumor_area_pct: float
    tumor_pixels: int
    total_pixels: int
    has_mask: bool
    skipped: bool


class RAGResult(BaseModel):
    llm_report: str
    source: str
    retrieved_docs: list
    language: str


class AnalysisResponse(BaseModel):
    classification: ClassificationResult
    segmentation: SegmentationResult
    clinical_report: dict
    rag_report: RAGResult
    images: dict
    inference_time_ms: float


class QARequest(BaseModel):
    question: str
    report_context: str
    class_name: str
    conversation_history: Optional[list] = []
    language: str = "en"


class TranslateRequest(BaseModel):
    text: str
    language: str


@app.get("/")
def root():
    return {
        "name": "X-Brain API v2.0",
        "status": "running",
        "startup_mode": "lazy" if not EAGER_LOAD_MODELS else "eager",
        "models": {
            "classifier": "loaded" if "classifier" in MODELS else "missing",
            "segmentor": (
                "disabled" if not ENABLE_SEGMENTOR else "loaded" if "segmentor" in MODELS else "missing"
            ),
        },
        "rag": "disabled" if not ENABLE_RAG else "ready" if _rag_index_exists() else "not_built",
        "features": [
            "classification",
            "gradcam",
            "segmentation",
            "rag_report",
            "qa",
            "translation",
        ],
    }


@app.get("/health")
def health():
    clf_ok = "classifier" in MODELS
    seg_ok = "segmentor" in MODELS
    return {
        "status": "ok" if clf_ok or not ENABLE_CLASSIFIER else "degraded",
        "classifier": clf_ok,
        "segmentor": seg_ok if ENABLE_SEGMENTOR else "disabled",
        "groq_llm": bool(os.getenv("GROQ_API_KEY")),
        "rag_enabled": ENABLE_RAG,
        "rag_index": _rag_index_exists() if ENABLE_RAG else False,
    }


@app.get("/languages")
def languages():
    if not ENABLE_RAG:
        return {"English": "en"}
    return _get_rag_module().get_supported_languages()


@app.post("/index/build")
def rebuild_index(force: bool = False):
    if not ENABLE_RAG:
        return {"success": False, "message": "RAG is disabled by configuration"}

    success = _get_rag_module().build_index(force=force)
    return {
        "success": success,
        "message": "Index built" if success else "Build failed or no PDFs found",
    }


@app.post("/qa")
def question_answer(req: QARequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not ENABLE_RAG:
        raise HTTPException(status_code=503, detail="RAG question answering is disabled by configuration.")

    answer = _get_rag_module().answer_question(
        question=req.question,
        report_context=req.report_context,
        class_name=req.class_name,
        conversation_history=req.conversation_history,
        language=req.language,
    )
    return {"answer": answer, "language": req.language}


@app.post("/translate")
def translate(req: TranslateRequest):
    if not ENABLE_RAG:
        return {"translated": req.text, "language": req.language}

    translated = _get_rag_module().translate_text(req.text, req.language)
    return {"translated": translated, "language": req.language}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...), language: str = "en"):
    classifier_model = _ensure_classifier_loaded()

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        img_rgb = read_image_from_bytes(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    t0 = time.time()

    classifier_module = _get_classifier_module()
    clf_result = classifier_module.classify(classifier_model, img_rgb)
    log.info("Classification: %s (%.2f%%)", clf_result["class_name"], clf_result["confidence"] * 100)

    heatmap_rgb, gradcam_overlay = classifier_module.get_gradcam_overlay(
        classifier_model,
        img_rgb,
        pred_index=clf_result["class_idx"],
    )

    segmentor_model, segmentor_device = _ensure_segmentor_loaded()
    skipped = not clf_result["has_tumor"] or segmentor_model is None

    if skipped:
        mask = np.zeros((224, 224), dtype=np.float32)
        seg_overlay = np.zeros((224, 224, 3), dtype=np.uint8)
        tumor_stats = {
            "tumor_area_pct": 0.0,
            "tumor_pixels": 0,
            "total_pixels": int(mask.size),
            "has_mask": False,
        }
    else:
        segmentor_module = _get_segmentor_module()
        mask = segmentor_module.segment(segmentor_model, img_rgb, segmentor_device)
        seg_overlay = segmentor_module.get_segmentation_overlay(img_rgb, mask)
        tumor_stats = segmentor_module.compute_tumor_stats(mask)

    report = generate_clinical_report(
        class_name=clf_result["class_name"],
        confidence=clf_result["confidence"],
        probabilities=clf_result["probabilities"],
        tumor_area_pct=tumor_stats["tumor_area_pct"],
        has_mask=tumor_stats["has_mask"],
        language=language,
    )

    if ENABLE_RAG:
        rag_result = _get_rag_module().generate_rag_report(
            class_name=clf_result["class_name"],
            confidence=clf_result["confidence"],
            tumor_area_pct=tumor_stats["tumor_area_pct"],
            has_mask=tumor_stats["has_mask"],
            probabilities=clf_result["probabilities"],
            language=language,
        )
    else:
        rag_result = {
            "llm_report": "RAG reporting is disabled in this deployment.",
            "source": "disabled",
            "retrieved_docs": [],
            "language": language,
        }

    inference_time = round((time.time() - t0) * 1000, 1)
    log.info("Total inference time: %s ms", inference_time)

    img_224 = cv2.resize(img_rgb, (224, 224))
    images = {
        "original": ndarray_to_base64(img_224),
        "gradcam_heatmap": ndarray_to_base64(heatmap_rgb),
        "gradcam_overlay": ndarray_to_base64(gradcam_overlay),
        "seg_mask": mask_to_base64(mask),
        "seg_overlay": ndarray_to_base64(seg_overlay),
    }

    return AnalysisResponse(
        classification=ClassificationResult(**clf_result),
        segmentation=SegmentationResult(
            tumor_area_pct=tumor_stats["tumor_area_pct"],
            tumor_pixels=tumor_stats["tumor_pixels"],
            total_pixels=tumor_stats["total_pixels"],
            has_mask=tumor_stats["has_mask"],
            skipped=skipped,
        ),
        clinical_report=report,
        rag_report=RAGResult(**rag_result),
        images=images,
        inference_time_ms=inference_time,
    )
