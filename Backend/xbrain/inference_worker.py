"""
inference_worker.py
Run a single X-Brain analysis in a short-lived subprocess so heavy TensorFlow
memory is released back to the OS after each request.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.classifier import build_classifier, classify, get_gradcam_overlay
from models.segmentor import (
    build_segmentor,
    compute_tumor_stats,
    get_segmentation_overlay,
    segment,
)
from utils.clinical_knowledge import generate_clinical_report
from utils.image_utils import mask_to_base64, ndarray_to_base64, read_image_from_bytes


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="en")
    parser.add_argument("--patient-id", default="")
    parser.add_argument("--clf-weights", required=True)
    parser.add_argument("--seg-weights", default="")
    args = parser.parse_args()

    raw = sys.stdin.buffer.read()
    if not raw:
        raise ValueError("Empty file uploaded.")

    img_rgb = read_image_from_bytes(raw)
    t0 = time.time()

    classifier_model = build_classifier(args.clf_weights)
    clf_result = classify(classifier_model, img_rgb)
    heatmap_rgb, gradcam_overlay = get_gradcam_overlay(
        classifier_model,
        img_rgb,
        pred_index=clf_result["class_idx"],
    )

    enable_segmentor = _env_flag("ENABLE_SEGMENTOR", False)
    skipped = True
    if enable_segmentor and clf_result["has_tumor"] and args.seg_weights:
        segmentor_model, segmentor_device = build_segmentor(args.seg_weights)
        mask = segment(segmentor_model, img_rgb, segmentor_device)
        seg_overlay = get_segmentation_overlay(img_rgb, mask)
        tumor_stats = compute_tumor_stats(mask)
        skipped = False
    else:
        mask = np.zeros((224, 224), dtype=np.float32)
        seg_overlay = np.zeros((224, 224, 3), dtype=np.uint8)
        tumor_stats = {
            "tumor_area_pct": 0.0,
            "tumor_pixels": 0,
            "total_pixels": int(mask.size),
            "has_mask": False,
        }

    report = generate_clinical_report(
        class_name=clf_result["class_name"],
        confidence=clf_result["confidence"],
        probabilities=clf_result["probabilities"],
        tumor_area_pct=tumor_stats["tumor_area_pct"],
        has_mask=tumor_stats["has_mask"],
        patient_id=args.patient_id or "N/A",
        language=args.language,
    )

    inference_time = round((time.time() - t0) * 1000, 1)
    img_224 = cv2.resize(img_rgb, (224, 224))
    payload = {
        "classification": clf_result,
        "segmentation": {
            "tumor_area_pct": tumor_stats["tumor_area_pct"],
            "tumor_pixels": tumor_stats["tumor_pixels"],
            "total_pixels": tumor_stats["total_pixels"],
            "has_mask": tumor_stats["has_mask"],
            "skipped": skipped,
        },
        "clinical_report": report,
        "rag_report": {
            "llm_report": "RAG reporting is disabled in this deployment.",
            "source": "disabled",
            "retrieved_docs": [],
            "language": args.language,
        },
        "images": {
            "original": ndarray_to_base64(img_224),
            "gradcam_heatmap": ndarray_to_base64(heatmap_rgb),
            "gradcam_overlay": ndarray_to_base64(gradcam_overlay),
            "seg_mask": mask_to_base64(mask),
            "seg_overlay": ndarray_to_base64(seg_overlay),
        },
        "inference_time_ms": inference_time,
    }
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write(str(exc))
        raise
