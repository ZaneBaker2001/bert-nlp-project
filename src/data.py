from __future__ import annotations
from datasets import load_dataset, DatasetDict
from typing import Optional


def get_dataset(name: Optional[str], text_col: str, label_col: str,
                train_split: Optional[str], validation_split: Optional[str],
                train_file: Optional[str], validation_file: Optional[str]) -> DatasetDict:
    if name:
        ds = load_dataset(name)
        train = train_split or "train"
        val = validation_split or "validation"
        assert train in ds and val in ds, f"Splits not found in {list(ds.keys())}"
        return DatasetDict({"train": ds[train], "validation": ds[val]})
    else:
        assert train_file and validation_file, "Provide train_file and validation_file for local CSVs"
        ds = load_dataset("csv", data_files={"train": train_file, "validation": validation_file})
        return ds
