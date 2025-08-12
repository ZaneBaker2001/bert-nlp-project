from __future__ import annotations
import os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = os.getenv("MODEL_DIR", "models/best")

class PredictIn(BaseModel):
    texts: List[str]
    top_k: Optional[int] = 1

class PredictOutItem(BaseModel):
    text: str
    predictions: List[tuple[str, float]]

app = FastAPI(title="BERT Text Classifier")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_DIR}

@app.post("/predict", response_model=List[PredictOutItem])
def predict(payload: PredictIn):
    inputs = tokenizer(payload.texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k=min(payload.top_k or 1, probs.shape[-1]), dim=-1)

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    results: List[PredictOutItem] = []
    for i, text in enumerate(payload.texts):
        labels = [id2label[int(ix)] for ix in top.indices[i].tolist()]
        scores = [float(s) for s in top.values[i].tolist()]
        results.append(PredictOutItem(text=text, predictions=list(zip(labels, scores))))
    return results
