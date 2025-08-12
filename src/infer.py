from __future__ import annotations
import argparse, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def predict(model_dir: str, texts: list[str], top_k: int = 1):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    results = []
    for i, text in enumerate(texts):
        labels = [id2label[int(ix)] for ix in top.indices[i].tolist()]
        scores = [float(s) for s in top.values[i].tolist()]
        results.append({"text": text, "predictions": list(zip(labels, scores))})
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--text", action="append", required=True)
    ap.add_argument("--top_k", type=int, default=1)
    args = ap.parse_args()

    out = predict(args.model_dir, args.text, args.top_k)
    from pprint import pprint
    pprint(out)
