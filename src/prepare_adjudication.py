import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT_DIR / "data" / "processed" / "processed_corpus.csv"
FAILURES_PATH = ROOT_DIR / "data" / "processed" / "model_failures.csv"
QUEUE_PATH = ROOT_DIR / "data" / "processed" / "adjudication_queue.csv"

def sync_test_set():
    print("Re-synchronizing test set indices...")
    # Load the original cleaned corpus (Exactly as in baselines/preprocessing)
    df = pd.read_csv(CORPUS_PATH)
    df['text'] = df['text'].astype(str)
    df = df[df['is_empty'] == 0].copy()
    
    # Re run the exact same split (seed 42 is critical here)
    # This gives us the exact same documents that are in 'test' dataset
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Load the 14 failures
    failures = pd.read_csv(FAILURES_PATH)
    
    # Map failures back to the real data using the index
    # The 'test_doc_X' ID corresponds to the row number in the test_df
    failure_indices = [int(x.split('_')[-1]) for x in failures['filename']]
    
    adjudication_data = test_df.iloc[failure_indices].copy()
    adjudication_data['minilm_pred'] = failures['minilm_pred'].values
    
    # Save the queue with filenames and text
    adjudication_data.to_csv(QUEUE_PATH, index=False)
    print(f"Successfully mapped {len(adjudication_data)} failures to real filenames.")
    print(f"Queue saved to: {QUEUE_PATH}")

if __name__ == "__main__":
    sync_test_set()