import os
import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm
import csv
from pathlib import Path

# Directory configuration
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_ROOT = ROOT_DIR / "data" / "raw" / "Train-v0"
OUTPUT_CSV = ROOT_DIR / "data" / "processed" / "processed_corpus.csv"

def extract_text(pdf_path):
    """
    Attempts to extract text content from a PDF file using the MuPDF engine.
    
    Args:
        pdf_path (Path): Absolute path to the PDF document.
        
    Returns:
        str: Extracted text or an empty string if the file is unreadable.
    """
    # Silence C-level MuPDF warnings (xref errors) for a cleaner terminal output
    fitz.TOOLS.mupdf_display_errors(False)
    
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception:
        # Catching generic exceptions to ensure the pipeline continues 
        # even if individual files are corrupted.
        return ""

def run_extraction():
    """
    Walks through the document taxonomy, extracts text from all PDFs, 
    and saves the resulting corpus to a structured CSV.
    """
    corpus_data = []
    
    if not DATA_ROOT.exists():
        print(f"Error: Raw data directory not found at {DATA_ROOT}")
        return

    # Identify document classes based on top-level folder names
    class_folders = [f for f in os.listdir(DATA_ROOT) if (DATA_ROOT / f).is_dir()]
    
    print(f"Found {len(class_folders)} document classes. Starting extraction...")

    for class_label in class_folders:
        class_path = DATA_ROOT / class_label
        
        # Recursively find all PDFs within the class and its regional subfolders
        files_to_process = []
        for root, _, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    files_to_process.append(os.path.join(root, file))

        print(f"Processing {class_label}: {len(files_to_process)} files")
        
        for file_path in tqdm(files_to_process, desc=f" {class_label}"):
            raw_text = extract_text(file_path)
            char_count = len(raw_text)
            
            # Heuristic: Documents with < 50 chars are flagged as likely scans 
            # or image-based PDFs requiring OCR.
            corpus_data.append({
                "filename": os.path.basename(file_path),
                "path": file_path,
                "label": class_label,
                "text": raw_text,
                "char_count": char_count,
                "is_empty": 1 if char_count < 50 else 0
            })

    # Export to CSV with strict quoting to handle technical characters in text
    df = pd.DataFrame(corpus_data)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(
        OUTPUT_CSV, 
        index=False, 
        quoting=csv.QUOTE_ALL, 
        escapechar='\\', 
        encoding='utf-8-sig'
    )
    
    print_extraction_summary(df)

def print_extraction_summary(df):
    """
    Outputs descriptive statistics of the extraction process for 
    inclusion in the dissertation methodology chapter.
    """
    total = len(df)
    empty_count = df['is_empty'].sum()
    
    print("\n" + "="*50)
    print("EXTRACTION PIPELINE SUMMARY")
    print("="*50)
    print(f"Total PDF Documents:   {total}")
    print(f"Searchable Documents:  {total - empty_count}")
    print(f"Non-Searchable/Scans:  {empty_count} ({(empty_count/total)*100:.2f}%)")
    print(f"Export Destination:    {OUTPUT_CSV}")
    print("="*50)
    
    # Identify which classes are most affected by the 'scan' problem
    print("\nPotential OCR Requirements by Class:")
    print(df.groupby('label')['is_empty'].sum().sort_values(ascending=False))

if __name__ == "__main__":
    run_extraction()