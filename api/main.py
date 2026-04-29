"""
DocuClassAI FastAPI backend.

Serves a two-tier cascading classification pipeline for UK statutory planning documents:
  Tier 1 - fine-tuned MiniLM-L12-H384 transformer (~98% of documents, ~140ms)
  Tier 2 - local Mistral-Nemo 12B adjudicator via Ollama (low-confidence and
            semantically ambiguous documents, routed by the confidence gate)
"""

import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import httpx
import joblib
import torch
import torch.nn.functional as F
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "docuclass_minilm")))
ENCODER_PATH = Path(os.getenv("ENCODER_PATH", str(BASE_DIR / "models" / "baselines" / "label_encoder.joblib")))
CLASS_DEFS_PATH = Path(os.getenv("CLASS_DEFS_PATH", str(BASE_DIR / "data" / "processed" / "class_definitions.json")))

# 0.80 is deliberately risk averse: calibrated on the validation set to minimise
# misclassification of Fire Statements and Environmental Statements, which carry
# direct regulatory consequence under UK planning law and the Building Safety Act 2022.
# Do not lower this value without re-running validation on your deployment corpus.
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))

# Documents with fewer extractable characters than this are treated as image based
# scans and routed to human review rather than force classified!!.
FIDELITY_MIN_CHARS: int = int(os.getenv("FIDELITY_MIN_CHARS", "150"))

ADJUDICATOR_URL: str = os.getenv("ADJUDICATOR_URL", "http://localhost:11434/api/generate")
ADJUDICATOR_MODEL: str = os.getenv("ADJUDICATOR_MODEL", "mistral-nemo")

# Hugging Face Hub repo containing the fine-tuned MiniLM weights.
# If MODEL_PATH does not exist locally, weights are downloaded from this repo
# automatically on first startup. Set to empty string to disable auto-download.
HF_MODEL_REPO: str = os.getenv("HF_MODEL_REPO", "Jeanmady/docuclass-minilm")

# Comma separated list of allowed CORS origins (default: Vite dev server)
CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

# The Environmental Statement / Biodiversity Survey conflict pair triggers Tier 2
# escalation regardless of confidence score. 
_CONFLICT_PAIR: frozenset[str] = frozenset({
    "Environmental statement",
    "Biodiversity survey and report",
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("docuclassai")

# ---------------------------------------------------------------------------
# Application state populated during startup, read only during requests
# ---------------------------------------------------------------------------

_state: dict = {}


def _ensure_model_weights() -> None:
    """
    Download fine-tuned MiniLM weights from Hugging Face Hub if not present locally.

    Called once at startup. If MODEL_PATH already contains the weights this is a
    no-op. If the directory is missing and HF_MODEL_REPO is set, snapshot_download
    pulls the full repository into MODEL_PATH. This allows a fresh clone to start
    the server without any manual model setup step.
    """
    if MODEL_PATH.exists() and any(MODEL_PATH.iterdir()):
        return

    if not HF_MODEL_REPO:
        raise RuntimeError(
            f"Model weights not found at {MODEL_PATH} and HF_MODEL_REPO is not set.\n"
            "Either place the model files manually or set HF_MODEL_REPO in .env."
        )

    logger.info(
        "Model weights not found locally. Downloading from Hugging Face Hub: %s",
        HF_MODEL_REPO,
    )
    logger.info("This only happens on first run — subsequent starts are instant.")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=HF_MODEL_REPO, local_dir=str(MODEL_PATH))
    logger.info("Model download complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download model weights if needed, then load everything into memory."""
    _ensure_model_weights()

    for path in (ENCODER_PATH, CLASS_DEFS_PATH):
        if not path.exists():
            raise RuntimeError(
                f"Required artefact missing at startup: {path}\n"
                "See README — 'Model Setup' section for details."
            )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Loading DocuClassAI on device: %s", device)

    _state["device"] = device
    _state["encoder"] = joblib.load(ENCODER_PATH)
    _state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_PATH)
    _state["model"] = (
        AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    )
    _state["model"].eval()

    with open(CLASS_DEFS_PATH, "r", encoding="utf-8") as fh:
        _state["class_definitions"] = json.load(fh)

    logger.info(
        "DocuClassAI ready — %d classes loaded, confidence threshold %.2f",
        len(_state["class_definitions"]),
        CONFIDENCE_THRESHOLD,
    )
    yield
    logger.info("DocuClassAI backend shutting down.")


app = FastAPI(
    title="DocuClassAI",
    description="Cascading hybrid classifier for UK statutory planning documents.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    device: str
    classes_loaded: int


class ClassificationResponse(BaseModel):
    label: str
    confidence: float
    tier: str
    escalation_reason: Optional[str] = None
    fidelity_status: str


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def extract_text_from_pdf(content: bytes) -> str:
    """
    Extract text from a PDF byte stream.

    PyMuPDF (fitz) is tried first: it handles complex layouts, CID-mapped fonts,
    and multi column documents better than pypdf. If PyMuPDF raises any exception
    (corrupted stream, unsupported encoding), pypdf is used as a fallback.
    Neither failure is fatal an empty string is returned and the fidelity gate
    will route the document to human review rather than producing a bad result.
    """
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception:
        logger.warning("PyMuPDF extraction failed, falling back to pypdf.")

    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            reader.decrypt("")
        return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        logger.warning("pypdf fallback also failed returning empty string.")
        return ""


def extract_text_from_docx(content: bytes) -> str:
    """Extract plain paragraph text from a DOCX byte stream."""
    doc = Document(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs)


def passes_fidelity_gate(text: str) -> bool:
    """
    Determine whether a document contains enough extractable text for reliable
    automated classification.

    Documents below FIDELITY_MIN_CHARS are treated as image-based scans (e.g.
    photographs of paper documents). These are routed to the human review queue
    rather than force classified. This is a deliberate safety requirement: Fire
    Statements disproportionately arrive as unreadable scans, and a misclassification
    in that category has direct consequences under the Building Safety Act 2022.

    Returns True if the document may proceed to classification, False otherwise.
    """
    return len(text.strip()) >= FIDELITY_MIN_CHARS


def head_tail_tokenize(text: str, tokenizer) -> dict[str, torch.Tensor]:
    """
    Tokenise text using the head-tail strategy for the 512-token window.

    The window reserves 2 tokens for [CLS] and [SEP], leaving 510 usable slots.
    These are split 255 head + 255 tail so the model sees the document opening
    (title, document type declaration) and the document closing (conclusions,
    author signatures, appendix headings). Truncating only from the tail would
    discard closing material that often disambiguates structurally similar report
    types (e.g. Environmental Statement vs. Biodiversity Survey).

    This 255/255 split was calibrated during training and must not be changed
    without re-running the full training and evaluation cycle.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= 510:
        input_ids = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
    else:
        head = tokens[:255]
        tail = tokens[-255:]
        input_ids = [tokenizer.cls_token_id] + head + tail + [tokenizer.sep_token_id]

    input_ids_t = torch.tensor([input_ids])
    attention_mask = torch.ones_like(input_ids_t)
    return {"input_ids": input_ids_t, "attention_mask": attention_mask}


async def adjudicate_with_mistral(filename: str, text: str) -> tuple[str, str]:
    """
    Call the local Mistral-Nemo adjudicator via Ollama for Tier 2 classification.

    The full class definitions from class_definitions.json are included in the
    prompt so Mistral can reason across the complete statutory taxonomy rather
    than guessing from name alone. The adjudicator URL and model name are
    configurable via environment variables for on-premise deployment.

    Returns (resolved_label, escalation_reason). On connection failure, returns
    ("ADJUDICATION_FAILED", reason_string) so the caller can surface a meaningful
    error to the frontend rather than crash the request.
    """
    definitions: dict = _state["class_definitions"]
    valid_labels = list(definitions.keys())
    labels_str = ", ".join(f"'{label}'" for label in valid_labels)
    snippet = f"{text[:1500]}\n[...]\n{text[-1500:]}"
    defs_block = "\n".join(f"- {k}: {v}" for k, v in definitions.items())

    prompt = (
        "[INST] You are a UK Statutory Planning Expert.\n"
        "TASK: Classify the document below into EXACTLY ONE of the following categories:\n"
        f"{labels_str}\n\n"
        "REFERENCE DEFINITIONS:\n"
        f"{defs_block}\n\n"
        f"DOCUMENT SNIPPET (File: {filename}):\n{snippet}\n\n"
        "DECISION RULE: Choose the most specific applicable category. "
        "If the document covers multiple environmental topics comprehensively "
        "with a Non-Technical Summary, it is an 'Environmental statement'.\n"
        "Response format: Output only the category name from the list. No explanation. [/INST]"
    )

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                ADJUDICATOR_URL,
                json={"model": ADJUDICATOR_MODEL, "prompt": prompt, "stream": False},
            )
        raw = response.json()["response"].strip().strip("'\"")

        # Match the returned string against the known taxonomy (partial match allowed
        # because Mistral sometimes includes trailing punctuation or minor paraphrasing)
        matched = next(
            (
                label for label in valid_labels
                if label.lower() in raw.lower() or raw.lower() in label.lower()
            ),
            None,
        )
        if matched:
            logger.info("Adjudicator resolved '%s' → '%s'", filename, matched)
            return matched, "low_confidence_or_conflict_pair"
        else:
            logger.warning("Adjudicator returned unrecognised label '%s' for '%s'", raw, filename)
            return raw, "low_confidence_or_conflict_pair"

    except Exception as exc:
        logger.error("Adjudicator call failed for '%s': %s", filename, exc)
        return "ADJUDICATION_FAILED", f"adjudicator_error: {type(exc).__name__}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe. Returns device in use and number of classes loaded."""
    return HealthResponse(
        status="ok",
        device=str(_state.get("device", "not_loaded")),
        classes_loaded=len(_state.get("class_definitions", {})),
    )


@app.post("/classify", response_model=ClassificationResponse, status_code=200)
async def classify(file: UploadFile = File(...)) -> ClassificationResponse:
    """
    Classify a planning document using the two-tier cascading pipeline.

    Accepts PDF or DOCX. Returns: predicted statutory class, confidence score,
    tier used (Tier 1 or Tier 2), escalation reason if applicable, and fidelity
    status (PASSED or FAILED).

    Documents routed by the fidelity gate return label HUMAN_REVIEW_REQUIRED
    and fidelity_status FAILED. These should not be auto-filed.
    """
    content = await file.read()
    filename = file.filename or "unknown"

    # --- Text extraction ---
    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(content)
    elif filename.lower().endswith(".docx"):
        try:
            text = extract_text_from_docx(content)
        except Exception as exc:
            logger.error("DOCX extraction failed for '%s': %s", filename, exc)
            raise HTTPException(status_code=422, detail="Could not extract text from DOCX.")
    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Submit PDF or DOCX.",
        )

    # --- Fidelity gate ---
    if not passes_fidelity_gate(text):
        logger.info(
            "Fidelity gate: '%s' routed to human review (char count %d < %d).",
            filename,
            len(text.strip()),
            FIDELITY_MIN_CHARS,
        )
        return ClassificationResponse(
            label="HUMAN_REVIEW_REQUIRED",
            confidence=0.0,
            tier="N/A",
            escalation_reason=(
                "Document appears to be an image-based scan with insufficient extractable text. "
                "Manual review required."
            ),
            fidelity_status="FAILED",
        )

    # --- Tier 1: MiniLM triage ---
    tokenizer = _state["tokenizer"]
    model = _state["model"]
    device = _state["device"]
    encoder = _state["encoder"]

    inputs = head_tail_tokenize(text, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0][pred_idx].item())

    tier1_label: str = encoder.inverse_transform([pred_idx])[0]
    is_conflict = tier1_label in _CONFLICT_PAIR

    logger.info(
        "Tier 1 — file: '%s' | label: '%s' | confidence: %.4f | conflict_pair: %s",
        filename,
        tier1_label,
        confidence,
        is_conflict,
    )

    # --- Tier 2: Mistral-Nemo adjudicator ---
    if confidence < CONFIDENCE_THRESHOLD or is_conflict:
        if is_conflict and confidence >= CONFIDENCE_THRESHOLD:
            escalation_reason = "conflict_class_pair"
        elif confidence < CONFIDENCE_THRESHOLD and not is_conflict:
            escalation_reason = "low_confidence"
        else:
            escalation_reason = "low_confidence_and_conflict_pair"

        logger.info(
            "Tier 2 escalation — file: '%s' | reason: %s", filename, escalation_reason
        )
        final_label, _ = await adjudicate_with_mistral(filename, text)

        return ClassificationResponse(
            label=final_label,
            confidence=round(confidence, 4),
            tier="Tier 2",
            escalation_reason=escalation_reason,
            fidelity_status="PASSED",
        )

    return ClassificationResponse(
        label=tier1_label,
        confidence=round(confidence, 4),
        tier="Tier 1",
        escalation_reason=None,
        fidelity_status="PASSED",
    )
