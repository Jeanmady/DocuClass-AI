"""
Corpus extraction pipeline.

Walks the raw training data directory, extracts text from every PDF, and writes
the result to a single CSV used by all downstream training and evaluation scripts.

PyMuPDF is the primary extractor. pypdf is used as a fallback. Documents that
yield fewer characters than FIDELITY_MIN_CHARS are flagged as low-fidelity scans
rather than silently discarded so that the training pipeline can report coverage.
"""

import csv
import os
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm

try:
    import fitz  # PyMuPDF
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False

# Directory configuration
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_ROOT = ROOT_DIR / "data" / "raw" / "Train-v0"
OUTPUT_CSV = ROOT_DIR / "data" / "processed" / "processed_corpus.csv"

# Documents with fewer characters than this are image-based scans. They are
# flagged with is_empty=1 rather than dropped, so the fidelity analysis in
# prepare_data.py can exclude them from the training set explicitly.
FIDELITY_MIN_CHARS: int = 150


def passes_fidelity_gate(text: str, min_chars: int = FIDELITY_MIN_CHARS) -> bool:
    """
    Determine whether an extracted text has sufficient content for reliable
    automated classification.

    Documents below min_chars are assumed to be image-based scans (e.g.
    photographed paper documents) where PyMuPDF and pypdf can extract no
    useful text. These must be routed to human review rather than
    force-classified. This requirement is especially important for Fire
    Statements, which disproportionately arrive as scans and carry direct
    regulatory consequence under the Building Safety Act 2022.

    Returns True if the document may proceed to classification, False if it
    should be flagged for human review.
    """
    return len(text.strip()) >= min_chars


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract text from a PDF file using PyMuPDF as the primary extractor,
    with pypdf as a fallback.

    PyMuPDF handles complex layouts, CID-mapped fonts, and multi-column
    documents more reliably than pypdf. If PyMuPDF is unavailable or raises
    an exception, pypdf is attempted. If both fail, an empty string is
    returned — the caller uses passes_fidelity_gate() to decide what to do
    with low-yield documents rather than crashing.
    """
    if _PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(str(pdf_path))
            text = "".join(page.get_text() for page in doc)
            doc.close()
            if text.strip():
                return text.strip()
        except Exception:
            pass  # Fall through to pypdf

    try:
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text.strip()
    except Exception:
        return ""


def run_extraction() -> None:
    """
    Walk DATA_ROOT, extract text from every PDF, and write processed_corpus.csv.

    Each row records: filename, path, label (class folder name), extracted text,
    character count, and is_empty flag. The is_empty flag is set when the document
    does not pass the fidelity gate — downstream scripts use this to filter scans.
    """
    corpus_data = []

    if not DATA_ROOT.exists():
        print(f"Error: Raw data directory not found at {DATA_ROOT}")
        return

    class_folders = [f for f in os.listdir(DATA_ROOT) if (DATA_ROOT / f).is_dir()]
    print(f"Found {len(class_folders)} classes. Starting extraction...")

    for class_label in class_folders:
        class_path = DATA_ROOT / class_label
        files_to_process: list[str] = []
        for root, _, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    files_to_process.append(os.path.join(root, file))

        print(f"Processing {class_label}: {len(files_to_process)} files")

        for file_path in tqdm(files_to_process, desc=f" {class_label}"):
            raw_text = extract_text_from_pdf(file_path)
            char_count = len(raw_text)

            corpus_data.append({
                "filename": os.path.basename(file_path),
                "path": file_path,
                "label": class_label,
                "text": raw_text,
                "char_count": char_count,
                "is_empty": 0 if passes_fidelity_gate(raw_text) else 1,
            })

    df = pd.DataFrame(corpus_data)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", encoding="utf-8-sig")
    _print_extraction_summary(df)


def _print_extraction_summary(df: pd.DataFrame) -> None:
    """Print a concise summary of extraction coverage to stdout."""
    total = len(df)
    empty_count = int(df["is_empty"].sum())
    print("\n" + "=" * 50)
    print("EXTRACTION PIPELINE SUMMARY")
    print("=" * 50)
    print(f"Total PDF Documents:   {total}")
    print(f"Searchable Documents:  {total - empty_count}")
    print(f"Non-Searchable/Scans:  {empty_count} ({(empty_count / total) * 100:.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    run_extraction()
