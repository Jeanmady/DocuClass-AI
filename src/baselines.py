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
    Loads the processed corpus and applies aggressive sanitisation to 
    ensure Scikit-learn vectorizers receive 100% string data.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Corpus not found at {file_path}. Run extraction first.")
    
    df = pd.read_csv(file_path)
    
    # 1. Force the 'text' column to string type and handle real NaN objects
    df['text'] = df['text'].astype(str)
    
    # 2. Replace common 'null' string indicators with actual NaNs for easy dropping
    # This catches "nan", "None", "NULL", and whitespace-only strings
    df['text'] = df['text'].replace(['nan', 'None', 'nan ', ' nan'], np.nan, regex=True)
    df = df.dropna(subset=['text'])
    
    # 3. Final filter: Ensure character count is > 50 and is_empty flag is 0
    # (Sometimes empty docs get a char_count of 3 or 4 due to metadata)
    clean_df = df[(df['is_empty'] == 0) & (df['text'].str.len() > 50)].copy()
    
    # One last safety check: ensure everything is still a string
    clean_df['text'] = clean_df['text'].apply(lambda x: str(x) if not isinstance(x, str) else x)
    
    print(f"Sanitisation complete. Training on {len(clean_df)} valid string documents.")
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
    Generates and exports normalized confusion matrices (percentages) 
    to highlight performance across imbalanced classes.
    """
    plt.figure(figsize=(18, 14))
    
    # Calculate raw confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # Normalise by row (True Labels)
    # use np.errstate to handle any classes with zero samples in the test set
    with np.errstate(all='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized) # Replace NaN with 0

    # Plot using the normalized data
    # fmt='.2f' shows proportions (e.g., 0.85). Use '.0%' for percentages (e.g., 85%)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 8}) # Smaller font to fit percentages
    
    plt.title(f"Normalized Confusion Matrix: {model_name}\n(Values represent Recall per class)")
    plt.ylabel('Ground Truth Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / f"cm_{model_name}_normalized.png"
    plt.savefig(save_path, dpi=300) 
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