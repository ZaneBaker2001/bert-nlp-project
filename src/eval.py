from __future__ import annotations
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate
from .config import Config
from .data import get_dataset
from .utils import console


def main(model_dir: str, cfg_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    if cfg_path:
        from .config import Config
        cfg = Config.load(cfg_path)
    else:
        # try to infer basic setup using ag_news
        class _Tmp:
            pass
        cfg = _Tmp()
        cfg.dataset = _Tmp()
        cfg.dataset.name = "ag_news"
        cfg.dataset.text_column = "text"
        cfg.dataset.label_column = "label"
        cfg.dataset.train_split = "train"
        cfg.dataset.validation_split = "test"
        cfg.dataset.train_file = None
        cfg.dataset.validation_file = None

    ds = get_dataset(cfg.dataset.name, cfg.dataset.text_column, cfg.dataset.label_column,
                     cfg.dataset.train_split, cfg.dataset.validation_split,
                     cfg.dataset.train_file, cfg.dataset.validation_file)

    tokenized = ds["validation"].map(lambda b: tokenizer(b[cfg.dataset.text_column], truncation=True), batched=True)
    tokenized = tokenized.remove_columns([c for c in tokenized.column_names if c not in [cfg.dataset.label_column, "input_ids", "token_type_ids", "attention_mask"]])

    import torch
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(tokenized.with_format("torch"), batch_size=32):
            labels = batch[cfg.dataset.label_column]
            batch = {k:v for k,v in batch.items() if k != cfg.dataset.label_column}
            out = model(**batch)
            preds.extend(out.logits.argmax(-1).cpu().numpy().tolist())
            refs.extend(labels.cpu().numpy().tolist())

    console.rule("[bold]Validation Metrics")
    print(classification_report(refs, preds, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(refs, preds))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
