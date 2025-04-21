from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
from dotenv import load_dotenv
from scripts.bluesky import initialize_client, get_last_message
from classes.architectures import CustomSentimentClassifier
import subprocess
import os
import json

# Load environment variables
load_dotenv()

# Load model
MODEL_PATH = os.environ.get("MODEL_PATH", "")
model = CustomSentimentClassifier.from_pretrained(MODEL_PATH)

# Init Bluesky client and get last message
client = initialize_client()
dernier_message = get_last_message(client)

# FastAPI app config
description_html = """
Sentiment analysis API based on latest messages from @lemonde.fr via Bluesky 🌐

<br><br>
<div style="display: flex; gap: 30px; align-items: center;">
    <img src="https://sesameworkshop.org/wp-content/uploads/2023/03/presskit_ss_bio_bert.png" width="180">
    <img src="https://storage.googleapis.com/media-newsinitiative/images/Le_Monde_Logo.original.png" width="140">
</div>
"""

app = FastAPI(
    title="BERT Sentiment Analysis",
    description=description_html,
    version="1.0",
    docs_url=None,
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        "message": "Welcome to the Sentiment Classification API 🎉",
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

# ======= Annotations endpoint =======
ANNOTATION_PATH = "annotations.jsonl"

class Annotation(BaseModel):
    text: str
    label: int

@app.post("/submit_annotation", tags=["Feedback"])
async def submit_annotation(data: Annotation):
    os.makedirs(os.path.dirname(ANNOTATION_PATH), exist_ok=True)
    with open(ANNOTATION_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(data.dict(), ensure_ascii=False) + "\n")
    return {"message": "Annotation saved ✅"}

# ======= Retrain endpoint =======
@app.post("/retrain_model", tags=["Training"])
async def retrain_model():
    try:
        result = subprocess.run(
            ["python", "retrain.py"], capture_output=True, text=True, check=True
        )
        return {"message": "Retraining complete ✅", "stdout": result.stdout}
    except subprocess.CalledProcessError as e:
        return {
            "message": "Retraining error ❌",
            "stdout": e.stdout,
            "stderr": e.stderr,
        }
