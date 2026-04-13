import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
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
    
    # Force the 'text' column to string type and handle real NaN objects
    df['text'] = df['text'].astype(str)
    
    # Replace common 'null' string indicators with actual NaNs for easy dropping
    df['text'] = df['text'].replace(['nan', 'None', 'nan ', ' nan'], np.nan, regex=True)
    df = df.dropna(subset=['text'])
    
    # Final filter: Ensure character count is > 50 and is_empty flag is 0
    clean_df = df[(df['is_empty'] == 0) & (df['text'].str.len() > 50)].copy()
    
    # Final safety check: ensure everything is a string
    clean_df['text'] = clean_df['text'].apply(lambda x: str(x) if not isinstance(x, str) else x)
    
    print(f"Sanitisation complete. Training on {len(clean_df)} valid string documents.")
    return clean_df

def run_baseline_comparison(df):
    """
    Performs 5-Fold Stratified Cross-Validation and a final Hold-out evaluation
    to establish a rigorous performance floor.
    """
    # Define Pipelines with N-Grams to capture context
    pipelines = [
        ("BoW_SVM", CountVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3))),
        ("TFIDF_SVM", TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3)))
    ]

    results = []

    for name, vectorizer in pipelines:
        print(f"\n--- Evaluating Pipeline: {name} ---")
        
        # Pre-process text for CV
        X = vectorizer.fit_transform(df['text'])
        y = df['label']
        
        # STRATIFIED CROSS-VALIDATION
        # use 5 folds to ensure every document is used in a test set once.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print("Running 5-Fold Stratified Cross-Validation...")
        
        cv_results = cross_validate(
            LinearSVC(class_weight='balanced', max_iter=5000, dual='auto'),
            X, y, 
            cv=skf, 
            scoring='f1_macro',
            n_jobs=-1 # Uses all CPU cores for speed
        )
        
        mean_f1 = cv_results['test_score'].mean()
        std_f1 = cv_results['test_score'].std()
        
        # FINAL HOLD OUT FOR VISUALISATION
        # still do a standard split to generate the Confusion Matrix
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
        )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        model = LinearSVC(class_weight='balanced', max_iter=5000, dual='auto')
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        holdout_acc = accuracy_score(y_test, y_pred)
        holdout_f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"CV Macro F1: {mean_f1:.4f} (+/- {std_f1:.4f})")
        print(f"Hold-out Accuracy: {holdout_acc:.4f}")

        results.append({
            "Model": name, 
            "CV_Mean_F1": mean_f1, 
            "CV_Std": std_f1, 
            "Holdout_Acc": holdout_acc
        })
        
        generate_visuals(y_test, y_pred, model.classes_, name)

    return pd.DataFrame(results)

def generate_visuals(y_true, y_pred, class_names, model_name):
    """
    Generates and exports normalized confusion matrices (percentages) 
    to highlight performance across imbalanced classes.
    """
    plt.figure(figsize=(18, 14))
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    with np.errstate(all='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 8})
    
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
    print(f"Dataset Loaded. Total samples: {len(dataset)}")
    
    summary = run_baseline_comparison(dataset)
    
    print("\n" + "="*50)
    print("FINAL BASELINE RESULTS (WITH CROSS-VALIDATION)")
    print("="*50)
    print(summary)