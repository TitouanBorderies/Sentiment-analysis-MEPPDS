import os
import json
import pandas as pd
from transformers import TrainingArguments
from dotenv import load_dotenv
import mlflow
from datetime import datetime
import shutil

from classes.trainer import CustomTrainer
from classes.architectures import CustomSentimentClassifier, tokenize_function
from scripts.data_processing import load_data
from classes.TweetDataset import TweetDataset

load_dotenv()

MODEL_PATH = os.environ.get("MODEL_PATH", "")
ANNOTATION_PATH = "annotations/annotations.jsonl"

def load_annotations(path=ANNOTATION_PATH):
    if not os.path.exists(path):
        print("Aucune annotation utilisateur trouvée.")
        return pd.DataFrame(columns=["text", "label"])
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    return pd.DataFrame(lines)

# Chargement des données
df_train, df_validation = load_data()
df_user_annotations = load_annotations()

if not df_user_annotations.empty:
    print(f"Ajout de {len(df_user_annotations)} annotations utilisateur au modèle.")

    # Tokenisation des annotations
    annotations_encodings = tokenize_function(df_user_annotations)
    annotations_dataset = TweetDataset(annotations_encodings, df_user_annotations["label"].tolist())

    # Charger modèle existant
    model = CustomSentimentClassifier.from_pretrained(MODEL_PATH)

    # Préparer le timestamp pour nom unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    retrained_output_dir = f"./models_retrain/model_{timestamp}"

    # Config entraînement
    training_args = TrainingArguments(
        output_dir=retrained_output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=20,
        logging_dir="./logs_retrain",
        logging_steps=1,
        save_steps=10,
        save_total_limit=1,
        disable_tqdm=False,
        report_to="none"
    )

    # MLflow setup
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("bert-sentiment-retrain")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=annotations_dataset,
        eval_dataset=None,
    )

    # Entraînement + log MLflow
    with mlflow.start_run():
        mlflow.log_param("num_train_epochs", training_args.num_train_epochs)
        mlflow.log_param("train_batch_size", training_args.per_device_train_batch_size)
        mlflow.log_param("eval_batch_size", training_args.per_device_eval_batch_size)
        mlflow.log_param("added_user_annotations", len(df_user_annotations))

        trainer.train()

        # Sauvegarde du modèle réentraîné dans un nouveau dossier avec horodatage
        trainer.save_model(retrained_output_dir)
        mlflow.log_artifacts(retrained_output_dir, artifact_path="model")

        # Écrase le dossier original avec le nouveau modèle
        if os.path.exists(MODEL_PATH):
            shutil.rmtree(MODEL_PATH)
        shutil.copytree(retrained_output_dir, MODEL_PATH)
        print(f"Modèle mis à jour sauvegardé dans : {MODEL_PATH}")
else:
    print("Aucune annotation utilisateur trouvée, aucune mise à jour nécessaire.")
