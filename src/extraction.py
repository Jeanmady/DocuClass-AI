import os
import pandas as pd
from tqdm import tqdm
import csv
from pathlib import Path
from pypdf import PdfReader  # <--- New library

# Directory configuration
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_ROOT = ROOT_DIR / "data" / "raw" / "Train-v0"
OUTPUT_CSV = ROOT_DIR / "data" / "processed" / "processed_corpus.csv"

def extract_text(pdf_path):
    """
    Extracts text content using pypdf.
    """
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text.strip()
    except Exception:
        return ""

def run_extraction():
    corpus_data = []
    
    if not DATA_ROOT.exists():
        print(f"Error: Raw data directory not found at {DATA_ROOT}")
        return

    class_folders = [f for f in os.listdir(DATA_ROOT) if (DATA_ROOT / f).is_dir()]
    print(f"Found {len(class_folders)} classes. Starting extraction...")

    for class_label in class_folders:
        class_path = DATA_ROOT / class_label
        files_to_process = []
        for root, _, files in os.walk(class_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    files_to_process.append(os.path.join(root, file))

        print(f"Processing {class_label}: {len(files_to_process)} files")
        
        for file_path in tqdm(files_to_process, desc=f" {class_label}"):
            raw_text = extract_text(file_path)
            char_count = len(raw_text)
            
            corpus_data.append({
                "filename": os.path.basename(file_path),
                "path": file_path,
                "label": class_label,
                "text": raw_text,
                "char_count": char_count,
                "is_empty": 1 if char_count < 50 else 0
            })

    df = pd.DataFrame(corpus_data)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL, escapechar='\\', encoding='utf-8-sig')
    print_extraction_summary(df)

def print_extraction_summary(df):
    total = len(df)
    empty_count = df['is_empty'].sum()
    print("\n" + "="*50)
    print("EXTRACTION PIPELINE SUMMARY")
    print("="*50)
    print(f"Total PDF Documents:   {total}")
    print(f"Searchable Documents:  {total - empty_count}")
    print(f"Non-Searchable/Scans:  {empty_count} ({(empty_count/total)*100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    run_extraction()