# BERT Classification

Train, evaluate, and serve a BERT text classifier with Hugging Face.

## Features
- Config-driven via YAML (dataset, labels, hyperparams)
- Trainer API with early stopping & best model saving
- Clean FastAPI server for production‑style inference
- Docker support & tests

## Directory Structure
```
bert-nlp-project/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ Makefile
├─ Dockerfile
├─ .env.example
├─ configs/
│  └─ config.yaml
├─ data/                 # (optional) local CSVs if not using HF datasets
├─ models/               # saved checkpoints
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ models.py
│  ├─ train.py
│  ├─ eval.py
│  ├─ infer.py
│  ├─ server.py
│  └─ utils.py
└─ tests/
   └─ test_smoke.py
```

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
```

## Custom Dataset (CSV)
Provide a CSV with columns text and label (string or int). Update configs/config.yaml to set dataset.name: null and dataset.train_file / dataset.validation_file.

## Docker 
To use Docker, run the following commands:
```
docker build -t bert-nlp .
docker run --rm -p 8000:8000 bert-nlp
```

## Tests 
To run tests, run the following command:
```
pytest -q
```



