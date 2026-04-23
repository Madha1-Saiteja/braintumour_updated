"""
utils/rag_pipeline.py
Production RAG pipeline for X-Brain.
"""

from __future__ import annotations

import os
import re
import json
import logging
import pickle
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

# ── Load .env ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

log = logging.getLogger("xbrain.rag")

# ── Optional dependencies ────────────────────────────────────────────────────
try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False
    log.warning("faiss-cpu not installed — RAG retrieval disabled")

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except ImportError:
    ST_OK = False
    log.warning("sentence-transformers not installed — embeddings disabled")

try:
    from pypdf import PdfReader
    PDF_OK = True
except ImportError:
    PDF_OK = False
    log.warning("pypdf not installed — PDF ingestion disabled")

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False
    log.warning("groq not installed — LLM report generation disabled")

try:
    from deep_translator import GoogleTranslator
    TRANSLATE_OK = True
except ImportError:
    TRANSLATE_OK = False
    log.warning("deep-translator not installed — translation disabled")


# ── Config ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / os.getenv("DOCS_DIR", "medical_docs")
INDEX_PATH = ROOT / os.getenv("FAISS_INDEX_PATH", "checkpoints/faiss_index.index")
META_PATH = ROOT / os.getenv("FAISS_META_PATH", "checkpoints/faiss_meta.pkl")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "4"))
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-70b-8192")
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Arabic": "ar",
    "Chinese": "zh-CN",
    "Japanese": "ja",
    "Portuguese": "pt",
    "Russian": "ru",
}

# ── Cached singletons ────────────────────────────────────────────────────────
_embed_model: Optional["SentenceTransformer"] = None
_index = None
_metadata = None


def _get_embed_model() -> Optional["SentenceTransformer"]:
    global _embed_model
    if _embed_model is None and ST_OK:
        log.info(f"Loading embedding model: {EMBED_MODEL}")
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


# ── PDF ingestion ────────────────────────────────────────────────────────────
def _extract_text_from_pdf(path: Path) -> str:
    if not PDF_OK:
        return ""
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text.strip())
    return " ".join(parts)


def _chunk_text(text: str, source: str) -> list[dict]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk_words = words[start:start + CHUNK_SIZE]
        chunk_text = " ".join(chunk_words)
        chunk_text = re.sub(r"\s+", " ", chunk_text).strip()

        if len(chunk_text) > 50:
            chunks.append({
                "text": chunk_text,
                "source": source,
                "word_start": start,
            })

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def load_and_chunk_pdfs() -> list[dict]:
    DOCS_DIR.mkdir(exist_ok=True)
    all_chunks = []
    pdf_files = list(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        log.info("No PDF files found in medical_docs/. Using fallback only.")
        return []

    for pdf_path in pdf_files:
        log.info(f"Ingesting: {pdf_path.name}")
        text = _extract_text_from_pdf(pdf_path)
        chunks = _chunk_text(text, pdf_path.name)
        all_chunks.extend(chunks)
        log.info(f"  → {len(chunks)} chunks from {pdf_path.name}")

    return all_chunks


# ── FAISS index management ───────────────────────────────────────────────────
def _index_hash(chunks: list[dict]) -> str:
    content = json.dumps([c["text"][:50] for c in chunks], sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()[:12]


def build_index(force: bool = False) -> bool:
    global _index, _metadata

    if not (FAISS_OK and ST_OK and PDF_OK):
        log.warning("Missing dependencies — FAISS index not built")
        return False

    if INDEX_PATH.exists() and META_PATH.exists() and not force:
        log.info("FAISS index already exists. Use force=True to rebuild.")
        return True

    chunks = load_and_chunk_pdfs()
    if not chunks:
        log.warning("No chunks to index.")
        return False

    embed = _get_embed_model()
    texts = [c["text"] for c in chunks]

    log.info(f"Encoding {len(texts)} chunks…")
    vecs = embed.encode(texts, show_progress_bar=True, batch_size=64)
    vecs = np.array(vecs, dtype="float32")
    faiss.normalize_L2(vecs)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    INDEX_PATH.parent.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    _index = index
    _metadata = chunks

    log.info(f"✅ FAISS index built: {len(chunks)} chunks, dim={dim}")
    return True


def _load_index():
    global _index, _metadata

    if _index is not None and _metadata is not None:
        return _index, _metadata

    if not (FAISS_OK and INDEX_PATH.exists() and META_PATH.exists()):
        return None, None

    _index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        _metadata = pickle.load(f)

    return _index, _metadata


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    index, metadata = _load_index()

    if index is None or not ST_OK:
        log.warning("RAG index or embedding model unavailable — retrieval skipped")
        return []

    embed = _get_embed_model()
    q_vec = embed.encode([query], batch_size=1)
    q_vec = np.array(q_vec, dtype="float32")
    faiss.normalize_L2(q_vec)

    scores, indices = index.search(q_vec, min(top_k, index.ntotal))
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            chunk = dict(metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results


# ── LLM helpers ──────────────────────────────────────────────────────────────
def _get_groq_client() -> Optional["Groq"]:
    if not GROQ_OK:
        return None

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        log.warning("GROQ_API_KEY not set — LLM generation disabled")
        return None

    return Groq(api_key=api_key)


def _build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No external medical literature retrieved. Rely on fallback summary."

    parts = []
    for i, c in enumerate(chunks, 1):
        src = c.get("source", "unknown")
        parts.append(f"[{i}] (Source: {src})\n{c['text']}")

    return "\n\n".join(parts)


SYSTEM_PROMPT = """You are X-Brain, an AI assistant specialized in neuro-oncology and brain MRI analysis.

RULES:
- Be precise, concise, and evidence-based.
- Recommend consulting a qualified neurologist/radiologist.
- Never make a definitive diagnosis.
- Use medically cautious language.
- Acknowledge uncertainty when needed.
- Never fabricate citations or statistics.
"""


# ── Fallback report ──────────────────────────────────────────────────────────
def _fallback_report(
    class_name: str,
    confidence: float,
    tumor_area_pct: float,
    has_mask: bool
) -> str:
    if class_name.lower() == "notumor":
        return """
**AI Classification Report — X-Brain**

**Prediction:** No Tumor Detected

No tumor class was detected in this MRI scan. Clinical correlation is still recommended if symptoms persist.

**Recommended Next Steps:**
- Correlate with symptoms and clinical history
- Consider follow-up MRI if symptoms continue
- Obtain specialist review if clinically indicated

⚠️ This is an AI-assisted screening result. Not a clinical diagnosis.
""".strip()

    area_note = (
        f"Tumor area occupies approximately {tumor_area_pct:.1f}% of the scan region."
        if has_mask else
        "Segmentation was not performed or no reliable tumor mask was produced."
    )

    return f"""
**AI Classification Report — X-Brain**

**Prediction:** {class_name.capitalize()}  
**Confidence:** {confidence:.1%}

**Tumor Area Assessment:** {area_note}

**Clinical Overview:** The AI model classified this MRI scan as {class_name}. This result should be reviewed by a qualified radiologist and neurologist together with symptoms, patient history, and additional imaging.

**Recommended Next Steps:**
- Neurology / neurosurgery consultation
- Correlation with contrast-enhanced MRI if needed
- Multidisciplinary review
- Biopsy or further workup as clinically indicated

**Note:** RAG-enhanced report unavailable or LLM not configured.

⚠️ This is an AI-assisted screening result. Not a clinical diagnosis.
""".strip()


# ── RAG report generation ────────────────────────────────────────────────────
def generate_rag_report(
    class_name: str,
    confidence: float,
    tumor_area_pct: float,
    has_mask: bool,
    probabilities: dict,
    language: str = "en",
) -> dict:
    if class_name.lower() == "notumor":
        llm_report = _fallback_report(class_name, confidence, tumor_area_pct, has_mask)

        if language != "en" and TRANSLATE_OK:
            try:
                translator = GoogleTranslator(source="en", target=language)
                llm_report = translator.translate(llm_report)
            except Exception as e:
                log.warning(f"Translation failed: {e}")

        return {
            "llm_report": llm_report,
            "source": "fallback",
            "retrieved_docs": [],
            "language": language,
        }

    query = (
        f"Brain tumor type: {class_name}. "
        f"Confidence: {confidence:.1%}. "
        f"Tumor area: {tumor_area_pct:.1f}%. "
        "Provide clinical overview, symptoms, diagnostic workup, treatment, and prognosis."
    )

    chunks = retrieve(query)
    context = _build_context(chunks)
    client = _get_groq_client()

    if client:
        prompt = f"""
AI Classification: {class_name} (Confidence: {confidence:.1%})
Tumor Area: {tumor_area_pct:.1f}% {"(segmentation performed)" if has_mask else "(no segmentation)"}
Class Probabilities: {json.dumps({k: f"{v:.1%}" for k, v in probabilities.items()}, indent=2)}

RETRIEVED MEDICAL LITERATURE:
{context[:3000]}

Generate a structured clinical AI report covering:
1. Brief clinical overview of {class_name}
2. Key symptoms to monitor
3. Recommended diagnostic workup
4. Standard treatment approaches
5. Prognosis summary
6. Immediate next steps

Use bold section headers.
Keep each section to 2–3 sentences.
End with an AI screening disclaimer.
"""
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.3,
            )
            llm_report = resp.choices[0].message.content.strip()
            source = "llm+rag"
        except Exception as e:
            log.error(f"LLM generation failed: {e}")
            llm_report = _fallback_report(class_name, confidence, tumor_area_pct, has_mask)
            source = "fallback"
    else:
        llm_report = _fallback_report(class_name, confidence, tumor_area_pct, has_mask)
        source = "fallback"

    if language != "en" and TRANSLATE_OK:
        try:
            translator = GoogleTranslator(source="en", target=language)
            llm_report = translator.translate(llm_report)
        except Exception as e:
            log.warning(f"Translation failed: {e}")

    return {
        "llm_report": llm_report,
        "source": source,
        "retrieved_docs": [
            {
                "source": c["source"],
                "score": c.get("score", 0),
                "snippet": c["text"][:200],
            }
            for c in chunks
        ],
        "language": language,
    }


# ── Question answering ───────────────────────────────────────────────────────
def _fallback_answer(question: str, class_name: str, chunks: list[dict]) -> str:
    if chunks:
        snippet = chunks[0]["text"][:400]
        return (
            f"Based on retrieved medical literature regarding {class_name}:\n\n"
            f"{snippet}\n\n"
            f"Please consult a qualified medical professional for specific clinical guidance."
        )

    return (
        f"I do not have enough retrieved context to answer: '{question}'.\n\n"
        f"Please consult a qualified neurologist or radiologist regarding {class_name}."
    )


def answer_question(
    question: str,
    report_context: str,
    class_name: str,
    conversation_history: list[dict] | None = None,
    language: str = "en",
) -> str:
    client = _get_groq_client()
    chunks = retrieve(f"{class_name} brain tumor {question}", top_k=3)
    lit_ctx = _build_context(chunks)

    if conversation_history is None:
        conversation_history = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history[-6:])

    user_msg = f"""
You are answering a question about an MRI brain tumor analysis report.

Use the report context as the primary source of truth.
Use retrieved literature only as supporting information.
Be direct, clear, and medically cautious.
Do not make a final diagnosis.
End with a brief recommendation to consult a qualified medical professional.

Current Report Context:
{report_context[:1200]}

Retrieved Medical Literature:
{lit_ctx[:800]}

Question:
{question}
"""
    messages.append({"role": "user", "content": user_msg})

    if client:
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=512,
                temperature=0.2,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"QA LLM failed: {e}")
            answer = _fallback_answer(question, class_name, chunks)
    else:
        answer = _fallback_answer(question, class_name, chunks)

    if language != "en" and TRANSLATE_OK:
        try:
            translator = GoogleTranslator(source="en", target=language)
            answer = translator.translate(answer)
        except Exception as e:
            log.warning(f"Translation failed: {e}")

    return answer


# ── Translation ──────────────────────────────────────────────────────────────
def translate_text(text: str, target_language: str) -> str:
    if target_language == "en" or not text.strip():
        return text

    if not TRANSLATE_OK:
        return text + "\n\n*(Translation unavailable — install deep-translator)*"

    try:
        translator = GoogleTranslator(source="auto", target=target_language)

        if len(text) <= 4500:
            return translator.translate(text)

        parts = [text[i:i + 4500] for i in range(0, len(text), 4500)]
        return " ".join(translator.translate(part) for part in parts)
    except Exception as e:
        log.error(f"Translation error: {e}")
        return text


def get_supported_languages() -> dict[str, str]:
    return SUPPORTED_LANGUAGES.copy()