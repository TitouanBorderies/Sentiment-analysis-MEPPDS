#/bin/bash

python3 scripts.train_flow.py
uvicorn app.no_ui:app --host 0.0.0.0 --port 8000
