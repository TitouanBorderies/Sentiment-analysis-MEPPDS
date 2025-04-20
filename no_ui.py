from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from dotenv import load_dotenv
from scripts.bluesky import initialize_client, get_last_message
from classes.architectures import CustomSentimentClassifier
import subprocess
import os
import json
from fastapi.middleware.cors import CORSMiddleware

# Charger les variables d'environnement
load_dotenv()

# Initialiser le modèle
MODEL_PATH = os.environ.get("MODEL_PATH", "")
model = CustomSentimentClassifier.from_pretrained(MODEL_PATH)

# Initialiser le client Bluesky
client = initialize_client()
dernier_message = get_last_message(client)

# Création de l'app FastAPI sans documentation publique
app = FastAPI(
    title="Analyse de sentiment avec BERT",
    description="API REST sans UI pour l’analyse de sentiments",
    version="1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def welcome():
    return {
        "message": "Bienvenue sur l'API de classification de sentiment 🎉",
        "model": "Custom BERT Sentiment Classifier"
    }

@app.get("/predict_last_message")
async def predict_last():
    sentiment = model.infer_sentiment(dernier_message)
    return {
        "message": dernier_message,
        "sentiment": sentiment
    }

@app.get("/predict_text")
async def predict_text(text: str = "La situation est tendue."):
    sentiment = model.infer_sentiment(text)
    return {
        "text": text,
        "sentiment": sentiment
    }

# ======= Soumission d'annotations utilisateur =======

ANNOTATION_PATH = "annotations/annotations.jsonl"  # Assurez-vous de définir un dossier

class Annotation(BaseModel):
    text: str
    label: int  # 0 ou 1

@app.post("/submit_annotation")
async def submit_annotation(data: Annotation):
    # Crée le dossier si nécessaire
    os.makedirs(os.path.dirname(ANNOTATION_PATH), exist_ok=True)

    # Ajoute l'annotation au fichier JSONL
    with open(ANNOTATION_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(data.dict(), ensure_ascii=False) + "\n")
    return {"message": "Annotation enregistrée ✅"}

# ======= Déclenchement du réentraînement =======

@app.post("/retrain_model")
async def retrain_model():
    try:
        result = subprocess.run(
            ["python", "retrain.py"],
            capture_output=True,
            text=True,
            check=True
        )
        return {
            "message": "Réentraînement terminé ✅",
            "stdout": result.stdout
        }
    except subprocess.CalledProcessError as e:
        return {
            "message": "Erreur pendant le réentraînement ❌",
            "stdout": e.stdout,
            "stderr": e.stderr
        }
