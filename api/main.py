from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import fitz

app = FastAPI()

# Enable CORS for React
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load model & encoder
model_path = "../models/docuclass_minilm"
encoder = joblib.load("../models/baselines/label_encoder.joblib")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Extract Text
    doc = fitz.open(stream=await file.read(), filetype="pdf")
    text = "".join([page.get_text() for page in doc])
    
    # Inference
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        
    label = encoder.inverse_transform([pred])[0]
    return {"filename": file.filename, "label": label}