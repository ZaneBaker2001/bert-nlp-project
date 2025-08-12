# BERT NLP Project (Classification)

Train, evaluate, and serve a BERT text classifier with Hugging Face.

## Features
- Config-driven via YAML (dataset, labels, hyperparams)
- Trainer API with early stopping & best model saving
- Clean FastAPI server for production‑style inference
- Docker support & tests

## Quickstart
```bash
# 1) Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Train (AG News by default)
python -m src.train configs/config.yaml

# 3) Evaluate
python -m src.eval models/best

# 4) Inference CLI
python -m src.infer models/best --text "Apple unveils new iPhone with AI features"

# 5) Serve API
uvicorn src.server:app --host 0.0.0.0 --port 8000
# curl example
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"texts":["Stocks rally as CPI cools"], "top_k":3}'
