PY ?= python3
# all configs are listed in scripts/train_parallel.sh

.PHONY: setup data train eval export bench all serve app test clean

setup:            ## install dependencies
	$(PY) -m pip install -r requirements.txt

data:             ## download + clean + split -> data/processed/
	$(PY) -m src.data.load --config configs/base.yaml

train: data       ## every experiment config, all seeds (parallel)
	bash scripts/train_parallel.sh

eval:             ## the only step that reads the test split
	$(PY) -m src.eval

export:
	$(PY) -m src.export

bench: export
	$(PY) -m serve.benchmark

all: train eval export bench   ## full pipeline: data -> models -> reports -> serving bundle

serve:            ## pricing API on :8000
	$(PY) -m uvicorn serve.app:app --port 8000

app:              ## Streamlit premium explorer
	$(PY) -m streamlit run serve/streamlit_app.py

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf artifacts mlflow.db serve/model_bundle logs
