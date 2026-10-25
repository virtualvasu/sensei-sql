VENV ?= .venv
PY   := $(VENV)/bin/python

# Stick to the machine's NVIDIA driver (570, CUDA <= 12.8): torch 2.10.0 is the newest
# cu128 build, and vllm 0.19.1 is the newest release pinned to it (Qwen3.5 support since 0.17).
TORCH_INDEX := https://download.pytorch.org/whl/cu128

# Offline flags from .env apply to every target except `download`.
include .env
export

.PHONY: venv install install-app download check test

venv:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

install: venv
	$(PY) -m pip install -e ".[gpu,app,download,dev]" --extra-index-url $(TORCH_INDEX)

install-app: venv
	$(PY) -m pip install -e ".[app,dev]"

download:
	HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 $(PY) scripts/download.py

check:
	$(PY) scripts/check_setup.py

test:
	$(PY) -m pytest -q
