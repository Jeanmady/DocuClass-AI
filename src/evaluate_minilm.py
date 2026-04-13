import torch
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import classification_report

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_PATH = ROOT_DIR / "data" / "processed" / "processed_dataset"
ENCODER_PATH = ROOT_DIR / "models" / "baselines" / "label_encoder.joblib"
MODEL_PATH = ROOT_DIR / "models" / "docuclass_minilm"
OUTPUT_CSV = ROOT_DIR / "data" / "processed" / "model_failures.csv"

def generate_failure_list():
    print("Loading model and test set...")
    # Load the 'test' split only
    ds = load_from_disk(DATASET_PATH)["test"]
    encoder = joblib.load(ENCODER_PATH)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()

    all_preds = []
    actuals = ds["label"]
    
    print(f"Running inference on {len(ds)} test samples...")
    for i in range(len(ds)):
        # use the tokenized IDs directly from the dataset
        inputs = {
            "input_ids": torch.tensor([ds[i]["input_ids"]]).to(device),
            "attention_mask": torch.tensor([ds[i]["attention_mask"]]).to(device)
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            all_preds.append(pred)

    # Convert to readable labels
    actual_labels = encoder.inverse_transform(actuals)
    pred_labels = encoder.inverse_transform(all_preds)

    # Create a results dataframe
    results = pd.DataFrame({
        "filename": [f"test_doc_{i}" for i in range(len(ds))], # Temporary ID
        "actual": actual_labels,
        "minilm_pred": pred_labels
    })

    # Save only the ones the model got WRONG
    failures = results[results['actual'] != results['minilm_pred']]
    failures.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*40)
    print("EVALUATION COMPLETE")
    print("="*40)
    print(f"Total Test Samples: {len(ds)}")
    print(f"Model Failures:     {len(failures)}")
    print(f"Failures saved to:  {OUTPUT_CSV}")
    print("="*40)
    print("\nTop Mistaken Classes:")
    print(failures['actual'].value_counts().head(5))

if __name__ == "__main__":
    generate_failure_list()