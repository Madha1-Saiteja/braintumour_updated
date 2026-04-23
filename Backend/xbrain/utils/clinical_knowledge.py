"""
utils/clinical_knowledge.py
Knowledge-based clinical report generator for X-Brain.
Provides structured, medically-grounded reports for each tumor class.
"""

import logging
from datetime import datetime

log = logging.getLogger("xbrain")

# ── Clinical Knowledge Base ───────────────────────────────────────────────────

CLINICAL_KB = {
    "glioma": {
        "tumor_type":         "Glioma",
        "urgency":            "HIGH",
        "urgency_icon":       "🔴",
        "who_classification": "WHO Grade II–IV (varies by subtype: Astrocytoma, Oligodendroglioma, GBM)",
        "description": (
            "Gliomas are primary brain tumors arising from glial cells (astrocytes, oligodendrocytes, "
            "or ependymal cells). They represent the most common malignant primary brain tumors in adults. "
            "High-grade gliomas (Grade III–IV), particularly Glioblastoma Multiforme (GBM), are aggressive "
            "with rapid progression and poor prognosis without treatment."
        ),
        "symptoms": [
            "Progressive headaches (often worse in the morning)",
            "Seizures (new-onset in adults)",
            "Focal neurological deficits (weakness, speech difficulty, vision changes)",
            "Cognitive changes, memory impairment",
            "Nausea and vomiting (signs of elevated ICP)",
            "Personality or behavioral changes",
        ],
        "diagnosis_methods": [
            "Contrast-enhanced MRI (T1, T2/FLAIR sequences) — gold standard",
            "MR Spectroscopy to assess tumor metabolism",
            "Stereotactic biopsy or surgical resection for tissue diagnosis",
            "Molecular profiling: IDH1/2 mutation, 1p/19q co-deletion, MGMT methylation",
            "PET scan for metabolic activity assessment",
            "Neuropsychological evaluation",
        ],
        "treatment_options": [
            "Maximal safe surgical resection (craniotomy)",
            "Radiation therapy (60 Gy in 30 fractions for GBM — Stupp protocol)",
            "Temozolomide (TMZ) chemotherapy — concurrent + adjuvant",
            "Tumor Treating Fields (TTFields) — Optune device for GBM",
            "Bevacizumab (anti-VEGF) for recurrent GBM",
            "Clinical trials: CAR-T cell therapy, oncolytic virotherapy",
            "Corticosteroids (dexamethasone) for edema management",
        ],
        "prognosis": {
            "GBM (Grade IV)":        "Median OS ~15 months with Stupp protocol; 5-year survival <10%",
            "Anaplastic (Grade III)": "Median OS 2–3 years; IDH-mutant subtype has better outcomes",
            "Low-grade (Grade II)":   "Median OS 5–15 years; IDH mutation and 1p/19q co-deletion are favorable",
            "Key Prognostic Factors": "IDH mutation status, MGMT methylation, extent of resection, age, KPS score",
        },
        "follow_up": "MRI every 2–3 months post-treatment; RANO criteria for response assessment",
    },

    "meningioma": {
        "tumor_type":         "Meningioma",
        "urgency":            "MODERATE",
        "urgency_icon":       "🟠",
        "who_classification": "WHO Grade I (benign, ~80%), Grade II (atypical, ~15%), Grade III (anaplastic, ~5%)",
        "description": (
            "Meningiomas arise from the meningothelial cells of the arachnoid layer of the meninges. "
            "They are the most common primary intracranial tumors overall (~36% of all brain tumors). "
            "Most are benign (WHO Grade I) and slow-growing. They are more prevalent in females (2:1 ratio) "
            "and risk increases with age. Many are incidentally discovered on imaging."
        ),
        "symptoms": [
            "Gradual onset headaches",
            "Seizures (if near cortex)",
            "Visual disturbances (if affecting optic pathway)",
            "Hearing loss or tinnitus (if near cranial nerves VIII)",
            "Focal neurological deficits (location-dependent)",
            "Cognitive changes, memory issues",
            "Many meningiomas are asymptomatic at diagnosis",
        ],
        "diagnosis_methods": [
            "Contrast-enhanced MRI — characteristic 'dural tail' sign",
            "CT scan (shows calcifications in ~20%)",
            "Cerebral angiography (for pre-operative embolization planning)",
            "Histopathological examination after resection",
            "Ki-67 proliferation index assessment",
            "PET-DOTATATE scan for somatostatin receptor imaging",
        ],
        "treatment_options": [
            "Watchful waiting with serial MRI (small, asymptomatic Grade I)",
            "Microsurgical resection — Simpson Grade I–III resection",
            "Stereotactic radiosurgery (SRS): Gamma Knife, CyberKnife for <3 cm tumors",
            "Fractionated radiotherapy for large or residual tumors",
            "Hydroxyurea (limited evidence for recurrent cases)",
            "Somatostatin analogs (experimental for recurrent cases)",
        ],
        "prognosis": {
            "WHO Grade I":   "Recurrence 7–25% at 10 years; near-normal life expectancy with gross total resection",
            "WHO Grade II":  "Recurrence 29–52% at 5 years; adjuvant radiotherapy recommended",
            "WHO Grade III": "Median OS 2–3 years; aggressive multimodal treatment required",
            "Key Factors":   "Extent of resection (Simpson grade), WHO grade, location, patient age",
        },
        "follow_up": "MRI at 3 months post-surgery, then annually for Grade I; more frequently for Grade II/III",
    },

    "pituitary": {
        "tumor_type":         "Pituitary Adenoma",
        "urgency":            "LOW-MODERATE",
        "urgency_icon":       "🔵",
        "who_classification": "WHO Classification: Pituitary Neuroendocrine Tumor (PitNET); micro (<10mm) vs macro (≥10mm)",
        "description": (
            "Pituitary adenomas are benign tumors arising from the anterior pituitary gland. "
            "They account for approximately 10–25% of all intracranial neoplasms. "
            "They are classified as functioning (hormone-secreting) or non-functioning. "
            "Functioning subtypes include prolactinomas (~40%), GH-secreting (acromegaly), "
            "ACTH-secreting (Cushing's disease), and TSH-secreting adenomas."
        ),
        "symptoms": [
            "Visual field defects (bitemporal hemianopia from optic chiasm compression)",
            "Headache (especially with macroadenomas)",
            "Hormonal hypersecretion syndromes (Cushing's, acromegaly, galactorrhea/amenorrhea)",
            "Hypopituitarism symptoms (fatigue, weight gain, sexual dysfunction)",
            "Pituitary apoplexy (sudden headache, visual loss — emergency)",
            "Cranial nerve palsies (III, IV, VI — cavernous sinus invasion)",
        ],
        "diagnosis_methods": [
            "MRI with gadolinium enhancement (dedicated pituitary protocol)",
            "Complete hormonal panel: prolactin, GH, IGF-1, ACTH, cortisol, TSH, FSH/LH",
            "Formal visual field testing (Humphrey perimetry)",
            "Dynamic tests: dexamethasone suppression test (Cushing's), GHRH stimulation",
            "Inferior petrosal sinus sampling (if Cushing's suspected)",
            "Ophthalmological evaluation",
        ],
        "treatment_options": [
            "Dopamine agonists (cabergoline, bromocriptine) — first-line for prolactinomas",
            "Transsphenoidal surgery (endoscopic) — standard for most macroadenomas",
            "Somatostatin analogs (octreotide, lanreotide) for GH-secreting tumors",
            "Pasireotide — second-generation SSA for Cushing's disease",
            "Radiation therapy (SRS or fractionated) for residual/recurrent disease",
            "Hormone replacement therapy for hypopituitarism",
        ],
        "prognosis": {
            "Prolactinoma":      "Excellent; >80% respond to cabergoline; surgical cure rate 80–90% for microadenomas",
            "Acromegaly (GH)":   "Good with treatment; remission ~50–80% post-surgery; SSAs help",
            "Cushing's Disease": "Variable; remission 65–90% post-surgery; monitoring for recurrence critical",
            "Non-functioning":   "Good with surgery; 10-year recurrence ~15% with total resection",
        },
        "follow_up": "MRI at 3–6 months post-treatment, then annually; hormonal panels every 6–12 months",
    },

    "notumor": {
        "tumor_type":         "No Tumor Detected",
        "urgency":            "NONE",
        "urgency_icon":       "🟢",
        "who_classification": "N/A — Normal brain MRI findings",
        "description": (
            "The AI analysis did not identify features consistent with a brain tumor in this MRI scan. "
            "The scan appears within normal limits for the assessed parameters. "
            "This result should be interpreted in the full clinical context, including patient history, "
            "symptoms, and neurological examination findings."
        ),
        "symptoms": [
            "No tumor-specific symptoms expected based on AI analysis",
            "Any ongoing neurological symptoms warrant further clinical evaluation",
            "Correlation with patient history is essential",
        ],
        "diagnosis_methods": [
            "Correlation with clinical history and neurological examination",
            "Review of any previous imaging for comparison",
            "Consider dedicated sequences if high clinical suspicion persists (DWI, SWI, perfusion MRI)",
            "Neurological consultation if symptoms are present",
            "EEG if seizures are a concern",
        ],
        "treatment_options": [
            "No tumor-specific treatment indicated based on current analysis",
            "Symptomatic management as clinically appropriate",
            "Follow-up MRI if clinically indicated",
        ],
        "prognosis": {
            "Current Status":  "No tumor identified — favorable finding",
            "Recommendation":  "Clinical follow-up and correlation with symptoms",
            "Note":            "AI analysis is a screening tool; clinical judgment remains paramount",
        },
        "follow_up": "Routine clinical follow-up; repeat MRI only if new or worsening symptoms develop",
    },
}

AREA_THRESHOLDS = {
    "minimal":  (0,   2,   "Minimal tumor burden (<2% of scan area). Microlesion; high precision follow-up imaging recommended."),
    "small":    (2,   8,   "Small tumor volume (2–8% of scan area). Early-stage assessment; multidisciplinary team review advised."),
    "moderate": (8,   20,  "Moderate tumor burden (8–20% of scan area). Significant involvement; detailed surgical planning warranted."),
    "large":    (20,  100, "Large tumor volume (>20% of scan area). Extensive disease; urgent neurosurgical consultation required."),
}

CONFIDENCE_LEVELS = {
    "very_high": (0.90, 1.01, "Very High", "The AI model is highly confident in this classification. Result is strongly reliable for screening purposes."),
    "high":      (0.75, 0.90, "High",      "High confidence prediction. Clinical validation recommended before diagnostic decisions."),
    "moderate":  (0.55, 0.75, "Moderate",  "Moderate confidence. Results should be interpreted cautiously. Expert radiological review strongly advised."),
    "low":       (0.00, 0.55, "Low",       "Low confidence — ambiguous features. Manual radiological review is essential. Do not base clinical decisions on this result."),
}


def generate_clinical_report(
    class_name: str,
    confidence: float,
    probabilities: dict,
    tumor_area_pct: float,
    has_mask: bool,
    patient_id: str = "N/A",
    language: str = "en",
) -> dict:
    """
    Generate a structured clinical report from classification + segmentation outputs.
    Returns a dict with all fields needed by the frontend.
    """
    import copy
    kb = copy.deepcopy(CLINICAL_KB.get(class_name.lower(), CLINICAL_KB["notumor"]))

    # Confidence label
    conf_label = "Low"
    conf_note  = CONFIDENCE_LEVELS["low"][3]
    for key, (lo, hi, label, note) in CONFIDENCE_LEVELS.items():
        if lo <= confidence < hi:
            conf_label = label
            conf_note  = note
            break

    # Area interpretation
    area_interp = "No tumor detected — segmentation not performed."
    if has_mask and class_name.lower() != "notumor":
        for key, (lo, hi, text) in AREA_THRESHOLDS.items():
            if lo <= tumor_area_pct < hi:
                area_interp = text
                break

    report = {
        **kb,
        "patient_id":          patient_id,
        "report_date":         datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "confidence_label":    conf_label,
        "confidence_note":     conf_note,
        "area_interpretation": area_interp,
        "disclaimer": (
            "⚠️ FOR RESEARCH & EDUCATIONAL USE ONLY. This AI-generated report is NOT a medical diagnosis. "
            "It must not be used as a substitute for professional medical advice, diagnosis, or treatment. "
            "Always consult a qualified radiologist and neurologist. AI analysis may contain errors. "
            "Clinical decisions must be made by licensed healthcare professionals only."
        ),
    }

    # ── Translate all text fields if language is not English ─────────────────
    if language != "en":
        try:
            from utils.rag_pipeline import translate_text

            report["description"]         = translate_text(report["description"], language)
            report["area_interpretation"] = translate_text(report["area_interpretation"], language)
            report["confidence_note"]     = translate_text(report["confidence_note"], language)
            report["disclaimer"]          = translate_text(report["disclaimer"], language)
            report["follow_up"]           = translate_text(report["follow_up"], language)
            report["symptoms"]            = [translate_text(s, language) for s in report["symptoms"]]
            report["diagnosis_methods"]   = [translate_text(s, language) for s in report["diagnosis_methods"]]
            report["treatment_options"]   = [translate_text(s, language) for s in report["treatment_options"]]
            report["prognosis"]           = {
                k: translate_text(v, language) for k, v in report["prognosis"].items()
            }
        except Exception as e:
            log.warning(f"Clinical report translation failed (falling back to English): {e}")

    return report