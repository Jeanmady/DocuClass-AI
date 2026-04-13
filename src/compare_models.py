import torch
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_PATH = ROOT_DIR / "data" / "processed" / "processed_dataset"
ENCODER_PATH = ROOT_DIR / "models" / "baselines" / "label_encoder.joblib"
MINILM_PATH = ROOT_DIR / "models" / "docuclass_minilm"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

def run_evaluation():
    print("Loading MiniLM model and test set...")
    # Load the 'test' split which the model never saw during training
    ds = load_from_disk(DATASET_PATH)["test"]
    encoder = joblib.load(ENCODER_PATH)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(MINILM_PATH).to(device)
    model.eval()

    mini_preds = []
    actuals = ds["label"]
    
    print(f"Running Inference on {len(ds)} test documents...")
    # Batch processing for speed
    for i in range(len(ds)):
        # Convert input_ids to tensor
        input_ids = torch.tensor([ds[i]["input_ids"]]).to(device)
        attention_mask = torch.tensor([ds[i]["attention_mask"]]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            mini_preds.append(prediction)

    # Convert to labels
    actual_labels = encoder.inverse_transform(actuals)
    pred_labels = encoder.inverse_transform(mini_preds)

    # Print Classification Report
    print("\n" + "="*50)
    print("FINAL MINILM PERFORMANCE (TEST SET)")
    print("="*50)
    print(classification_report(actual_labels, pred_labels))

    # Generate Normalized Confusion Matrix
    generate_normalized_cm(actual_labels, pred_labels, encoder.classes_)

def generate_normalized_cm(y_true, y_pred, labels):
    plt.figure(figsize=(18, 14))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    with np.errstate(all='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', 
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 8})
    
    plt.title("MiniLM: Normalized Recall Heatmap (Test Set)")
    plt.ylabel('Ground Truth')
    plt.xlabel('MiniLM Prediction')
    plt.tight_layout()
    
    plt.savefig(FIGURES_DIR / "cm_minilm_final.png", dpi=300)
    print(f"\nHeatmap saved to {FIGURES_DIR / 'cm_minilm_final.png'}")

if __name__ == "__main__":
    run_evaluation()