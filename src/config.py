from __future__ import annotations
import dataclasses, yaml, os
from dataclasses import dataclass

@dataclass
class ModelCfg:
    pretrained_name: str = "bert-base-uncased"
    num_labels: int = 4

@dataclass
class TrainerCfg:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    fp16: bool = True
    logging_steps: int = 50
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    early_stopping_patience: int = 2

@dataclass
class DatasetCfg:
    name: str | None = "ag_news"
    text_column: str = "text"
    label_column: str = "label"
    train_split: str | None = "train"
    validation_split: str | None = "test"
    train_file: str | None = None
    validation_file: str | None = None

@dataclass
class Config:
    seed: int = 42
    output_dir: str = "models"
    model: ModelCfg = ModelCfg()
    trainer: TrainerCfg = TrainerCfg()
    dataset: DatasetCfg = DatasetCfg()

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        def merge(dc_cls, d):
            obj = dc_cls()
            for k, v in (d or {}).items():
                setattr(obj, k, v)
            return obj
        return Config(
            seed=raw.get("seed", 42),
            output_dir=raw.get("output_dir", "models"),
            model=merge(ModelCfg, raw.get("model")),
            trainer=merge(TrainerCfg, raw.get("trainer")),
            dataset=merge(DatasetCfg, raw.get("dataset")),
        )
