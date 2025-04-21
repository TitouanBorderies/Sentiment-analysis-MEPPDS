from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from dotenv import load_dotenv
from scripts.bluesky import initialize_client, get_message
from classes.architectures import CustomSentimentClassifier
from fastapi.middleware.cors import CORSMiddleware
from utils.filter_annotations import filter_and_save_clean_annotations
import subprocess
import os
import json

# Load environment variables
load_dotenv()

# Load model
MODEL_PATH = os.environ.get("MODEL_PATH", "")
model = CustomSentimentClassifier.from_pretrained(MODEL_PATH)

# Init Bluesky client
client = initialize_client()
dernier_message = get_message(client)

# FastAPI app without public documentation
app = FastAPI(
    title="BERT Sentiment Analysis",
    description="REST API without public docs",
    version="1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def welcome():
    return {
        "message": "Bienvenue sur l'API de classification de sentiment 🎉 hehe",
        "model": "Custom BERT Sentiment Classifier"
    }

@app.get("/predict_last_message")
async def predict_last():
    sentiment = model.infer_sentiment(dernier_message)
    return {"message": dernier_message, "sentiment": sentiment}

@app.get("/predict_text")
async def predict_text(text: str = "La situation est tendue."):
    sentiment = model.infer_sentiment(text)
    return {"text": text, "sentiment": sentiment}

# ======= Annotations submission =======
ANNOTATION_PATH = "annotations/annotations.jsonl"

class Annotation(BaseModel):
    text: str
    label: int

@app.post("/submit_annotation")
async def submit_annotation(data: Annotation):
    os.makedirs(os.path.dirname(ANNOTATION_PATH), exist_ok=True)
    with open(ANNOTATION_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(data.dict(), ensure_ascii=False) + "\n")
    filter_and_save_clean_annotations()
    return {"message": "Annotation saved and filtered ✅"}

# ======= Retraining endpoint =======
@app.post("/retrain_model")
async def retrain_model():
    try:
        result = subprocess.run(
            ["python3", "scripts/retrain.py"],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONPATH": os.getcwd()}  # Ajoute la racine du projet au PYTHONPATH

        )
        return {"message": "Retraining complete ✅", "stdout": result.stdout}
    except subprocess.CalledProcessError as e:
        return {
            "message": "Retraining error ❌",
            "stdout": e.stdout,
            "stderr": e.stderr,
        }

# ======= Get last Bluesky titles =======
@app.get("/get_last_titles")
async def get_last_titles():
    client = initialize_client()
    titres = []
    for i in range(10):
        try:
            titre = get_message(client, position=i)
            titres.append(titre)
        except Exception:
            break
    return {"titres": titres}
