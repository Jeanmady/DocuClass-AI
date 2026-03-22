import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from datasets import load_from_disk
import evaluate
import numpy as np
import joblib
from pathlib import Path

# Directory and Path Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_PATH = ROOT_DIR / "data" / "processed" / "processed_dataset"
ENCODER_PATH = ROOT_DIR / "models" / "baselines" / "label_encoder.joblib"
MODEL_OUT = ROOT_DIR / "models" / "docuclass_minilm"
LOGS_DIR = ROOT_DIR / "outputs" / "training_logs"

class FocalLoss(nn.Module):
    """
    Implements Alpha-Weighted Focal Loss. 
    Manual implementation to bypass MPS-specific weighted cross-entropy bugs.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # This is our weight tensor

    def forward(self, inputs, targets):
        # 1. Calculate standard cross entropy without weights first (reduction='none' is stable)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. Calculate probabilities
        pt = torch.exp(-ce_loss)
        
        # 3. Apply Focal term
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss
        
        # 4. Apply Alpha weights manually if provided
        if self.alpha is not None:
            # Ensure alpha is on the same device as the loss
            self.alpha = self.alpha.to(inputs.device)
            # Gather the weight corresponding to each target class in the batch
            batch_weights = self.alpha[targets]
            loss = loss * batch_weights
            
        return loss.mean()

class WeightedTrainer(Trainer):
    """
    Custom Trainer that correctly handles device placement for the Focal Loss weights.
    """
    def __init__(self, alpha_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Move weights to the correct device immediately
        self.loss_fct = FocalLoss(alpha=alpha_weights, gamma=2.0)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Ensure the loss function knows which device the model is on
        self.loss_fct.to(model.device)
        
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    """
    Calculates Macro F1-Score as the primary success metric to ensure 
    performance is measured equitably across imbalanced classes.
    """
    f1_metric = evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1_metric.compute(predictions=predictions, references=labels, average="macro")

def calculate_class_weights(dataset):
    """
    Computes balanced class weights using the inverse frequency method.
    Formula: total_samples / (num_classes * class_samples)
    """
    labels = dataset["label"]
    counts = np.bincount(labels)
    total = len(labels)
    n_classes = len(counts)
    
    weights = total / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float)

def run_training():
    """
    Initialises the model and orchestrates the fine-tuning process 
    on the Apple Silicon backend.
    """
    # Load data and classification labels
    ds = load_from_disk(DATASET_PATH)
    encoder = joblib.load(ENCODER_PATH)
    num_labels = len(encoder.classes_)

    # Calculate weights for the 'Alpha' parameter in Focal Loss
    print("Calculating class-aware weights for Focal Loss...")
    alpha_weights = calculate_class_weights(ds["train"])

    # Model and Tokenizer Setup
    model_nm = "microsoft/MiniLM-L12-H384-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_nm)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_nm, 
        num_labels=num_labels
    )

    # Training Configuration for Local/On-Premise environments
    training_args = TrainingArguments(
        output_dir=str(LOGS_DIR),
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        dataloader_num_workers=0 # Optimised for macOS process spawning
    )

    trainer = WeightedTrainer(
        alpha_weights=alpha_weights,
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"Executing training on: {training_args.device}")
    trainer.train()

    # Save the final fine-tuned model and tokenizer for deployment phase
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"Success: Model persisted to {MODEL_OUT}")

if __name__ == "__main__":
    run_training()