import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

# Path configuration using project root detection
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_corpus.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "figures"

# Ensure output directory exists for exported analysis visuals
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_sanitised_data(file_path):
    """
    Loads the corpus and applies strict type filtering to prevent 
    vectorizer crashes caused by empty PDF artifacts or CSV corruption.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Missing processed corpus at {file_path}")

    df = pd.read_csv(file_path)
    
    # Force text to string and replace 'nan' artifacts with true nulls for dropping
    df['text'] = df['text'].astype(str)
    df = df.replace('nan', np.nan)
    df = df.dropna(subset=['text'])

    # Exclude documents flagged as scans (0 words) or corrupt during extraction
    return df[df['char_count'] >= 50].copy()

def perform_error_analysis(df):
    """
    Trains a high-performance SVM analyzer and decomposes its failures
    to identify semantic overlaps between document classes.
    """
    # Split maintains the 188:1 imbalance ratio for realistic error detection
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )

    # Use n-grams (1,3) to capture technical planning terminology
    print("Extracting features and training analyzer...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Balanced weights force the model to 'care' about minority class mistakes
    model = LinearSVC(class_weight='balanced', max_iter=5000, dual='auto')
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    # 1. Class Performance Table
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    print("\n" + "="*50)
    print("PER-CLASS METRICS (Ranked by Difficulty)")
    print("="*50)
    # Isolate individual classes from summary stats
    class_stats = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    print(class_stats[['precision', 'recall', 'f1-score', 'support']].sort_values(by='f1-score'))

    # 2. Confusion Mapping (Top Mistaken Pairs)
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    mistakes = results_df[results_df['Actual'] != results_df['Predicted']]

    print("\n" + "="*50)
    print("TOP SEMANTIC CONFUSIONS (Actual -> Predicted)")
    print("="*50)
    if not mistakes.empty:
        print(mistakes.groupby(['Actual', 'Predicted']).size().sort_values(ascending=False).head(10))
    
    # 3. Export Visual Confusion Matrix
    export_visual_analysis(y_test, y_pred, model.classes_)

def export_visual_analysis(y_true, y_pred, labels):
    """
    Creates a high-resolution heatmap for the dissertation 
    to highlight clusters of classification failure.
    """
    plt.figure(figsize=(16, 12))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 'Reds' colormap is used to make misclassifications (false positives) visually striking
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=labels, yticklabels=labels)
    
    plt.title("Error Analysis: SVM Classification Confusions")
    plt.ylabel('Ground Truth Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / "error_analysis_heatmap.png")
    print(f"\nHeatmap exported to {OUTPUT_DIR / 'error_analysis_heatmap.png'}")

if __name__ == "__main__":
    print(f"Starting Error Analysis from Root: {ROOT_DIR}")
    data = load_sanitised_data(DATA_PATH)
    perform_error_analysis(data)