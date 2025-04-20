#/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH  # Ajoute la racine du projet au PYTHONPATH
python3 scripts/train_flow.py
uvicorn api_folder.no_ui:app --host 0.0.0.0 --port 8000
