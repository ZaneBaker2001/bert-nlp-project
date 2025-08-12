from __future__ import annotations
from transformers import AutoModelForSequenceClassification

def build_model(pretrained_name: str, num_labels: int):
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_name, num_labels=num_labels
    )
