#!/bin/bash
python3 train.py  # optional: only if you want to retrain before launch
uvicorn api:app --host "0.0.0.0" --port 8000
