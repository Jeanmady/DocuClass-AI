# DocuClassAI

Cascading hybrid classifier for UK statutory planning documents. Classifies planning application PDFs into 23 statutory categories using a two tier on-premise pipeline with no cloud API dependency.

---

## System Overview

DocuClassAI uses a cascading architecture designed for on-premise deployment at local planning authorities under GDPR constraints.

**Tier 1 — MiniLM Triage:** A fine-tuned `microsoft/MiniLM-L12-H384-uncased` transformer (22M parameters) handles approximately 98% of documents at ~140ms per document. Documents are tokenised using a head-tail strategy (255 head + 255 tail tokens) to capture both the document title and its conclusions within the 512-token window.

**Tier 2 — Mistral-Nemo Adjudicator:** The 2% of documents where Tier 1 confidence falls below 0.80, or where the predicted class belongs to the Environmental Statement / Biodiversity Survey conflict pair, are escalated to a locally-hosted Mistral-Nemo 12B model via Ollama. This adjudicator receives the full statutory class definitions and reasons zero-shot over the document.

**Fidelity Gate:** Before either tier runs, documents are checked for extractable text. Image based scans (common for Fire Statements) are routed to a human review queue rather than force classified.

All classification runs locally. No document content leaves the deployment machine.

---

## Requirements

- **Python:** 3.13 (pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/):** dependency manager — install via `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Node.js:** 18+ (for the React frontend)
- **Apple Silicon recommended:** the MPS backend is used automatically on Apple Silicon. CPU fallback is supported but will be slower for Tier 1 inference.
- **Ollama:** required for Tier 2 adjudication — install from [ollama.com](https://ollama.com)
- **Disk space:** ~8 GB for the Mistral-Nemo 12B model; ~500 MB for MiniLM weights

---

## Installation

```bash
git clone https://github.com/Jeanmady/DocuClass-AI.git
cd DocuClass-AI
uv sync
```

`uv sync` reads `uv.lock` to produce a byte for byte reproducible environment. No other setup is needed for the Python backend.

For the React frontend:

```bash
cd web_app
npm install
```

---

## Model Setup

### Tier 1 — Fine-tuned MiniLM (automatic)

The fine tuned weights are hosted on Hugging Face Hub. **No manual setup is needed** on first startup, `uv run serve` detects that the weights are missing and downloads them automatically:

```
INFO  Model weights not found locally. Downloading from Hugging Face Hub: Jeanmady/docuclass-minilm
INFO  This only happens on first run — subsequent starts are instant.
```

The download is ~130 MB and takes roughly 30–60 seconds on a standard connection. After that, the weights are cached in `models/docuclass_minilm/` and subsequent starts are instant.

The `label_encoder.joblib` and tokenizer config files are committed directly to the repository and require no separate download.

### Tier 2 — Mistral-Nemo via Ollama

```bash
ollama pull mistral-nemo
```

Ollama runs as a background service — if it is already running, you do not need to start it again.

---

## Configuration

Copy `.env.example` to `.env` and adjust for your environment:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/docuclass_minilm` | Path to fine tuned MiniLM weights |
| `ENCODER_PATH` | `models/baselines/label_encoder.joblib` | Path to label encoder |
| `CLASS_DEFS_PATH` | `data/processed/class_definitions.json` | Statutory class definitions for the adjudicator |
| `CONFIDENCE_THRESHOLD` | `0.80` | Minimum Tier 1 confidence before Tier 2 escalation. Calibrated on the validation set. Do not lower without re running validation. |
| `FIDELITY_MIN_CHARS` | `150` | Minimum extractable characters before a document is flagged as a scan and routed to human review |
| `ADJUDICATOR_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint for Mistral-Nemo |
| `ADJUDICATOR_MODEL` | `mistral-nemo` | Ollama model name |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins for the React frontend |

---

## Running the System

**Start the API backend:**

```bash
uv run serve
```

The backend starts at `http://localhost:8000`. Logs print to stdout including classification decisions, Tier 2 escalations, and fidelity gate rejections.

**Start the React frontend (separate terminal):**

```bash
cd web_app
npm run dev
```

The dashboard opens at `http://localhost:5173`.

---

## API Reference

### `GET /health`

Liveness probe. Returns the compute device in use and number of classes loaded.

```json
{
  "status": "ok",
  "device": "mps",
  "classes_loaded": 23
}
```

### `POST /classify`

Classify a planning document.

**Request:** `multipart/form-data` with a `file` field containing a PDF or DOCX.

**Response:**

```json
{
  "label": "Design and Access Statement",
  "confidence": 0.9741,
  "tier": "Tier 1",
  "escalation_reason": null,
  "fidelity_status": "PASSED"
}
```

| Field | Description |
|---|---|
| `label` | Predicted statutory class name, or `HUMAN_REVIEW_REQUIRED` for scan documents |
| `confidence` | Tier 1 softmax probability for the predicted class (0–1) |
| `tier` | `Tier 1`, `Tier 2`, or `N/A` (fidelity failure) |
| `escalation_reason` | `low_confidence`, `conflict_class_pair`, `low_confidence_and_conflict_pair`, or `null` |
| `fidelity_status` | `PASSED` or `FAILED` |

### `GET /manifest`

Not yet implemented. Use the **Export CSV** button in the dashboard to download a manifest of the current session's results.

---

## Architecture Overview

```
Document Input (PDF / DOCX)
         │
         ▼
 ┌───────────────┐
 │ Text Extraction│  PyMuPDF primary → pypdf fallback
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ Fidelity Gate │  char count < 150 → HUMAN_REVIEW_REQUIRED
 └───────┬───────┘
         │ PASSED
         ▼
 ┌───────────────────────┐
 │ Head-Tail Tokenisation│  255 head + 255 tail tokens
 └───────────┬───────────┘
             │
             ▼
 ┌───────────────────────┐
 │  MiniLM Tier 1 Triage │  confidence threshold 0.80
 └───────────┬───────────┘
             │
     ┌───────┴────────┐
     │ confidence      │ conflict pair or
     │ >= 0.80 AND     │ confidence < 0.80
     │ not conflict    │
     ▼                 ▼
 Tier 1 Result  ┌──────────────────────┐
                │ Mistral-Nemo Tier 2  │  Ollama, local, zero shot
                └──────────┬───────────┘
                           │
                           ▼
                      Tier 2 Result
                           │
                           ▼
                  FastAPI → React Dashboard
```

---

## Project Structure

```
DocuClass-AI/
├── api/
│   ├── main.py              FastAPI application: classification pipeline
│   └── cli.py               Entry point for `uv run serve`
├── src/
│   ├── extraction.py        Training corpus extraction (PyMuPDF + pypdf)
│   ├── prepare_data.py      Head-tail tokenisation, stratified split, HF dataset
│   ├── train_transformer.py MiniLM fine tuning with Alpha-Weighted Focal Loss
│   ├── evaluate_minilm.py   Test set evaluation and confusion matrix
│   ├── llm_adjuticator.py   Batch Mistral adjudication over model failure cases
│   ├── prepare_adjudication.py  Builds the adjudication queue CSV
│   ├── baselines.py         BoW-SVM and TF-IDF-SVM baseline experiments
│   ├── compare_models.py    Side by- ide metric comparison across all models
│   ├── error_analysis.py    Qualitative analysis of remaining errors
│   └── final_results.py     Final accuracy/F1 reporting
├── web_app/
│   └── src/App.jsx          React dashboard: upload, results table, CSV export
├── data/
│   └── processed/
│       └── class_definitions.json   Statutory definitions for 23 classes (adjudicator input)
├── models/
│   ├── docuclass_minilm/    Fine tuned MiniLM weights (git ignored, obtain separately)
│   └── baselines/           Label encoder (git ignored, obtain separately)
├── outputs/
│   └── figures/             Confusion matrices and evaluation plots
├── .env.example             Template for environment variable configuration
├── .python-version          Python 3.13 pin
├── pyproject.toml           Project metadata and dependencies
└── uv.lock                  Locked dependency versions 
```

---

## Reproducing Results

`uv.lock` cryptographically pins every dependency version. To validate the environment matches the one used for training:

```bash
uv sync
uv run python -c "import torch; print(torch.__version__)"
```

To retrain from scratch (requires the raw training corpus: see supervisor for access under NDA):

```bash
# 1. Extract text from raw PDFs
uv run python src/extraction.py

# 2. Tokenise, encode labels, build HuggingFace dataset
uv run python src/prepare_data.py

# 3. Fine-tune MiniLM with Alpha-Weighted Focal Loss
uv run python src/train_transformer.py

# 4. Evaluate on held-out test set
uv run python src/evaluate_minilm.py

# 5. Run Mistral adjudication over model failures
uv run python src/llm_adjuticator.py
```

---

## Acknowledgements

- **MiniLM:** Wang et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.* NeurIPS 2020.
- **Focal Loss:** Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV 2017.
- **Training corpus:** UK statutory planning documents provided by industry partner under NDA. Not included in this repository.
