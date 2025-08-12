.PHONY: train eval serve test fmt

train:
	python -m src.train configs/config.yaml

eval:
	python -m src.eval models/best

serve:
	uvicorn src.server:app --host 0.0.0.0 --port 8000

test:
	pytest -q
