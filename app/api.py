from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from joblib import load

# Load model and tokenizer
MODEL_PATH = "bert_model.joblib"
MODEL_NAME = "bert-base-uncased"  # Ensure this matches your training setup

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = load(MODEL_PATH)
model.eval()

# Define sentiment labels
label_map = {0: "negative", 1: "neutral", 2: "positive"}

# Define request structure
class TextRequest(BaseModel):
    text: str

# Create FastAPI instance
app = FastAPI(
    title="Sentiment Analysis API",
    description="API pour analyser le sentiment d’un texte avec un modèle BERT entraîné 🧠",
)

@app.get("/", tags=["Welcome"])
def read_root():
    return {"message": "Bienvenue sur l'API de classification de sentiment avec BERT."}

@app.post("/predict", tags=["Prediction"])
def predict_sentiment(request: TextRequest):
    # Tokenize input
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        sentiment = label_map[prediction]

    return {"text": request.text, "sentiment": sentiment}
