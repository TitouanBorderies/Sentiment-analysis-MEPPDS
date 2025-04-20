#/bin/bash

python3 train_flow.py
uvicorn api_folder.api:app --host "0.0.0.0"