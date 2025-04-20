from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html
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

# Création de l'app FastAPI
description_html = """
API de détection de sentiment à partir des derniers messages de @lemonde.fr via Bluesky 🌐

<br><br>
<div style="display: flex; gap: 30px; align-items: center;">
    <img src="https://sesameworkshop.org/wp-content/uploads/2023/03/presskit_ss_bio_bert.png" width="180">
    <img src="https://storage.googleapis.com/media-newsinitiative/images/Le_Monde_Logo.original.png" width="140">
</div>
"""

app = FastAPI(
    title="Analyse de sentiment avec BERT",
    description=description_html,
    version="1.0",
    docs_url=None,
    openapi_url="/openapi.json",  # ✅ Ceci réactive /openapi.json
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔓 Pour tests. Remplace par ton frontend en prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request):
    root_url = str(request.base_url)
    return get_swagger_ui_html(
        openapi_url=f"{root_url}openapi.json", title="Custom API docs"
    )


@app.get("/", tags=["Welcome"])
async def welcome():
    return {
        "message": "Bienvenue sur l'API de classification de sentiment 🎉",
        "model": "Custom BERT Sentiment Classifier",
    }


@app.get("/predict_last_message", tags=["Predict"])
async def predict_last():
    sentiment = model.infer_sentiment(dernier_message)
    return {"message": dernier_message, "sentiment": sentiment}


@app.get("/predict_text", tags=["Predict"])
async def predict_text(text: str = "La situation est tendue."):
    sentiment = model.infer_sentiment(text)
    return {"text": text, "sentiment": sentiment}


# ======= NOUVEAU ENDPOINT: soumission d'annotations utilisateur =======

ANNOTATION_PATH = "annotations.jsonl"


class Annotation(BaseModel):
    text: str
    label: int  # 0 ou 1, selon ton format de labels


@app.post("/submit_annotation", tags=["Feedback"])
async def submit_annotation(data: Annotation):
    os.makedirs(os.path.dirname(ANNOTATION_PATH), exist_ok=True)
    with open(ANNOTATION_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(data.dict(), ensure_ascii=False) + "\n")
    return {"message": "Annotation enregistrée ✅"}


# ======= NOUVEAU ENDPOINT: déclenchement du réentraînement =======


@app.post("/retrain_model", tags=["Training"])
async def retrain_model():
    try:
        result = subprocess.run(
            ["python", "retrain.py"], capture_output=True, text=True, check=True
        )
        return {"message": "Réentraînement terminé ✅", "stdout": result.stdout}
    except subprocess.CalledProcessError as e:
        return {
            "message": "Erreur pendant le réentraînement ❌",
            "stdout": e.stdout,
            "stderr": e.stderr,
        }
