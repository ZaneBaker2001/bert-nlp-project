from __future__ import annotations
import sys, os
from typing import List
from transformers import (AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback,
                          DataCollatorWithPadding)
import evaluate
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .config import Config
from .utils import set_seed, ensure_dir, console
from .models import build_model
from .data import get_dataset


def main(cfg_path: str):
    cfg = Config.load(cfg_path)
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_name, use_fast=True)

    ds = get_dataset(cfg.dataset.name, cfg.dataset.text_column, cfg.dataset.label_column,
                     cfg.dataset.train_split, cfg.dataset.validation_split,
                     cfg.dataset.train_file, cfg.dataset.validation_file)

    # Infer label mapping for string labels
    if ds["train"].features[cfg.dataset.label_column].dtype == "string":
        labels = sorted(set(ds["train"][cfg.dataset.label_column]))
        label2id = {l:i for i,l in enumerate(labels)}
        id2label = {i:l for l,i in label2id.items()}
        def map_labels(batch):
            batch[cfg.dataset.label_column] = [label2id[x] for x in batch[cfg.dataset.label_column]]
            return batch
        ds = ds.map(map_labels, batched=True)
        num_labels = len(labels)
    else:
        num_labels = int(max(ds["train"][cfg.dataset.label_column])) + 1
        id2label = {i:str(i) for i in range(num_labels)}
        label2id = {v:k for k,v in id2label.items()}

    def tokenize(batch):
        return tokenizer(batch[cfg.dataset.text_column], truncation=True)

    ds = ds.map(tokenize, batched=True, remove_columns=[c for c in ds["train"].column_names if c not in [cfg.dataset.label_column]])
    data_collator = DataCollatorWithPadding(tokenizer)

    metric_acc = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

    model = build_model(cfg.model.pretrained_name, num_labels)
    model.config.id2label = id2label
    model.config.label2id = label2id

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.trainer.num_train_epochs,
        per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.trainer.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        warmup_ratio=cfg.trainer.warmup_ratio,
        fp16=cfg.trainer.fp16,
        logging_steps=cfg.trainer.logging_steps,
        evaluation_strategy=cfg.trainer.eval_strategy,
        save_strategy=cfg.trainer.save_strategy,
        load_best_model_at_end=cfg.trainer.load_best_model_at_end,
        metric_for_best_model=cfg.trainer.metric_for_best_model,
        greater_is_better=cfg.trainer.greater_is_better,
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.trainer.early_stopping_patience)]
    )

    console.rule("[bold green]Training")
    trainer.train()

    console.rule("[bold green]Saving best model")
    best_dir = os.path.join(cfg.output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml")
