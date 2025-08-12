from __future__ import annotations
import os, random, numpy as np, torch
from rich.console import Console

console = Console()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(False)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
