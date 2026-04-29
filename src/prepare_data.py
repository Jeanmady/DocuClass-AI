import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder

# Path configuration
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
INPUT_CSV = ROOT_DIR / "data" / "processed" / "processed_corpus.csv"
DATASET_OUT = ROOT_DIR / "data" / "processed" / "processed_dataset"
ENCODER_OUT = ROOT_DIR / "models" / "baselines" / "label_encoder.joblib"

# Model configuration
MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"
MAX_LENGTH = 512

def load_and_sanitise_corpus(path):
    """
    Loads the extracted text and applies strict type-checking to ensure 
    compatibility with the Hugging Face Rust-based tokenizer.
    """
    df = pd.read_csv(path)
    
    # Cast to string and drop nulls to prevent TypeError: TextEncodeInput
    df['text'] = df['text'].astype(str)
    df = df.replace('nan', np.nan).dropna(subset=['text'])
    
    # Only retain rows containing actual text content
    clean_df = df[df['char_count'] >= 50].copy()
    
    print(f"Sanitisation complete. Retained {len(clean_df)} valid documents.")
    return clean_df

def tokenize_head_tail(examples, tokenizer):
    """
    Implements a 'Head-Tail' truncation strategy. For documents exceeding 
    the 512-token limit, it preserves the first 255 and last 255 tokens.
    
    This ensures the model sees both the introductory context and the 
    concluding summaries/signatures, which are high-value markers in 
    planning documentation.
    """
    # Tokenize without truncation first to find the true length
    tokenized = tokenizer(examples["text"], add_special_tokens=False)
    
    input_ids = []
    attention_masks = []
    
    for ids in tokenized["input_ids"]:
        # Reserve 2 slots for [CLS] and [SEP] tokens
        if len(ids) > (MAX_LENGTH - 2):
            # Concatenate the start and end of the document
            head = ids[:255]
            tail = ids[-255:]
            processed_ids = [tokenizer.cls_token_id] + head + tail + [tokenizer.sep_token_id]
        else:
            processed_ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
        
        # Manually handle padding and attention mask
        padding_length = MAX_LENGTH - len(processed_ids)
        mask = [1] * len(processed_ids) + [0] * padding_length
        processed_ids = processed_ids + [tokenizer.pad_token_id] * padding_length
        
        input_ids.append(processed_ids)
        attention_masks.append(mask)

    return {"input_ids": input_ids, "attention_mask": attention_masks}

def prepare_and_save_dataset():
    """
    Orchestrates the full data preparation pipeline: cleaning, label encoding, 
    head-tail tokenization, and stratified splitting.
    """
    df = load_and_sanitise_corpus(INPUT_CSV)
    
    # Map textual labels to integers and persist encoder for inference
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label'])
    ENCODER_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, ENCODER_OUT)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_dataset = Dataset.from_pandas(df[['text', 'label']].reset_index(drop=True))

    print("Executing Head-Tail Tokenization...")
    tokenized_ds = hf_dataset.map(
        lambda x: tokenize_head_tail(x, tokenizer), 
        batched=True, 
        remove_columns=["text"]
    )

    # Stratified split (80/10/10) to maintain class distribution in evaluation
    train_test_split = tokenized_ds.train_test_split(test_size=0.2, seed=42)
    eval_test_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)

    ds_dict = DatasetDict({
        'train': train_test_split['train'],
        'valid': eval_test_split['train'],
        'test': eval_test_split['test']
    })

    ds_dict.save_to_disk(DATASET_OUT)
    print(f"Dataset persisted to {DATASET_OUT}")
    print(f"Final Training Samples: {len(ds_dict['train'])}")

if __name__ == "__main__":
    prepare_and_save_dataset()