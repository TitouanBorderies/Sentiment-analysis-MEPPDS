from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from joblib import load
from dotenv import load_dotenv
from bluesky import initialize_client, get_last_message
from architectures import CustomSentimentClassifier

# Charger les variables d'environnement
load_dotenv()

# Initialiser le modèle
# Attention le path est hardcodé 
MODEL_PATH = "/home/onyxia/work/Sentiment-analysis-MEPPDS/prod_model"
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
    docs_url=None
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/proxy/8000/openapi.json",  # 👈 AJUSTÉ POUR ONYXIA
        title="Custom API docs"
    )

@app.get("/", tags=["Welcome"])
async def welcome():
    return {
        "message": "Bienvenue sur l'API de classification de sentiment 🎉",
        "model": "Custom BERT Sentiment Classifier",
        "version": "1.0"
    }

@app.get("/predict_last_message", tags=["Predict"])
async def predict_last():
    sentiment = model.infer_sentiment(dernier_message)
    return {
        "message": dernier_message,
        "sentiment": sentiment
    }

@app.get("/predict_text", tags=["Predict"])
async def predict_text(text: str = "La situation est tendue."):
    sentiment = model.infer_sentiment(text)
    return {
        "text": text,
        "sentiment": sentiment
    }
