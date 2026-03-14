import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Directory Management
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_corpus.csv"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

# Ensure output directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean_data(file_path):
    """
    Loads the processed corpus and performs final sanitisation for 
    traditional machine learning vectorizers.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Corpus not found at {file_path}. Run extraction first.")
    
    df = pd.read_csv(file_path)
    
    # Standardise text column to string type to prevent vectorizer type errors
    df['text'] = df['text'].astype(str)
    
    # Remove null artifacts and entries flagged as empty/scanned during extraction
    df = df[df['text'].str.strip().lower() != "nan"]
    clean_df = df[df['is_empty'] == 0].copy()
    
    return clean_df

def run_baseline_comparison(df):
    """
    Trains and evaluates multiple baseline models to establish a 
    performance floor for the document classification task.
    """
    # Stratified split ensures class proportions are maintained across 188:1 imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )

    # N-gram range (1, 3) allows the model to capture technical phrases 
    # such as 'Environmental Impact Assessment' rather than just individual words.
    pipelines = [
        ("BoW_SVM", CountVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3))),
        ("TFIDF_SVM", TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3)))
    ]

    results = []

    for name, vectorizer in pipelines:
        print(f"Executing Pipeline: {name}")
        
        # Feature Extraction
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # LinearSVC with balanced class weights to penalise minority class errors
        model = LinearSVC(class_weight='balanced', max_iter=5000, dual='auto')
        model.fit(X_train_vec, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results.append({"Model": name, "Accuracy": acc, "Macro F1": f1})
        print(f"{name} Result -> Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
        
        generate_visuals(y_test, y_pred, model.classes_, name)

    return pd.DataFrame(results)

def generate_visuals(y_true, y_pred, class_names, model_name):
    """
    Generates and exports confusion matrices to the project's output directory.
    """
    plt.figure(figsize=(15, 12))
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / f"cm_{model_name}.png"
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    print(f"Project Root: {ROOT_DIR}")
    dataset = load_and_clean_data(DATA_PATH)
    print(f"Dataset Loaded. Training on {len(dataset)} samples.")
    
    summary = run_baseline_comparison(dataset)
    
    print("\n" + "="*30)
    print("BASELINE PERFORMANCE SUMMARY")
    print("="*30)
    print(summary)