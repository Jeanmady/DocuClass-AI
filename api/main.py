from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pypdf import PdfReader
from docx import Document
import io
from pathlib import Path

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Paths setup
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "docuclass_minilm"
ENCODER_PATH = BASE_DIR / "models" / "baselines" / "label_encoder.joblib"

# Load model to GPU (MPS) or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"DocuClassAI Backend Loading on: {device}")

encoder = joblib.load(ENCODER_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    text = ""
    
    try:
        # 1. Multi-format Extraction
        if file.filename.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(content))
            # Handle encryption
            if reader.is_encrypted:
                try:
                    reader.decrypt("") # Attempt empty password
                except:
                    return {"label": "ERROR: PDF Encrypted", "confidence": 0, "tier": "N/A"}
            
            for page in reader.pages:
                text += page.extract_text() or ""
                
        elif file.filename.endswith(".docx"):
            doc = Document(io.BytesIO(content))
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            return {"label": "ERROR: Unsupported Format", "confidence": 0, "tier": "N/A"}
        
        # 2. Inference
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()
            
        label = encoder.inverse_transform([pred_idx])[0]
        
        # 3. Adjudication Logic (The "Original Method")
        # Flag if low confidence OR if it is one of the 'Semantic Conflict' classes
        is_conflict = label in ["Environmental statement", "Biodiversity survey and report"]
        needs_adjudication = confidence < 0.80 or is_conflict
        
        print(f"File: {file.filename} | Label: {label} | Conf: {confidence:.2f}")
        
        return {
            "label": label, 
            "confidence": round(confidence, 4),
            "tier": "Tier 2 (Expert)" if needs_adjudication else "Tier 1 (Triage)"
        }
        
    except Exception as e:
        print(f"Error processing {file.filename}: {e}")
        return {"label": "ERROR: Could not process", "confidence": 0, "tier": "N/A"}