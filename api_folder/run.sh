#/bin/bash

python3 train_flow.py
uvicorn app.api:app --host 0.0.0.0 --port 8000
