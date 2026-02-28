#!/bin/bash


rm -rf .venv

uv venv --python 3.12

source .venv/bin/activate

uv pip install -r requirements.txt

PYTHONPATH=. uvicorn main:app --host 0.0.0.0 --port 8000 --reload
