import os
import json
import pandas as pd
from transformers import TrainingArguments
from dotenv import load_dotenv
import mlflow

from classes.trainer import CustomTrainer
from classes.architectures import CustomSentimentClassifier, tokenize_function
from scripts.data_processing import load_data
from classes.TweetDataset import TweetDataset

load_dotenv()

ANNOTATION_PATH = "annotations/annotations.jsonl"

def load_annotations(path=ANNOTATION_PATH):
    """Charge les nouvelles annotations envoyées par les utilisateurs."""
    if not os.path.exists(path):
        print("Aucune annotation utilisateur trouvée.")
        return pd.DataFrame(columns=["text", "label"])
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    return pd.DataFrame(lines)

# Chargement des données
df_train, df_validation = load_data()
df_user_annotations = load_annotations()

# Si des annotations utilisateur sont présentes
if not df_user_annotations.empty:
    print(f"Ajout de {len(df_user_annotations)} annotations utilisateur au modèle.")

    # Tokenisation des nouvelles annotations
    annotations_encodings = tokenize_function(df_user_annotations)
    annotations_dataset = TweetDataset(annotations_encodings, df_user_annotations["label"].tolist())

    # Initialiser modèle pré-existant
    model = CustomSentimentClassifier.from_pretrained("prod_model")  # Charger le modèle existant

    # Arguments d'entraînement (plus léger que le training initial)
    training_args = TrainingArguments(
        output_dir="./models_retrain",
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

    # Setup MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("bert-sentiment-retrain")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=annotations_dataset,  # Seulement les annotations nouvelles
        eval_dataset=None,  # Pas nécessaire d'évaluer ici, mais à ajouter si besoin
    )

    # Enregistrer les paramètres du modèle dans MLflow
    with mlflow.start_run():
        mlflow.log_param("num_train_epochs", training_args.num_train_epochs)
        mlflow.log_param("train_batch_size", training_args.per_device_train_batch_size)
        mlflow.log_param("eval_batch_size", training_args.per_device_eval_batch_size)
        mlflow.log_param("added_user_annotations", len(df_user_annotations))

        # Entraînement du modèle uniquement sur les nouvelles annotations
        trainer.train()

        # Sauvegarde du modèle mis à jour
        model_path = os.path.join(training_args.output_dir, "updated_model")
        trainer.save_model(model_path)
        mlflow.log_artifacts(model_path, artifact_path="model")

else:
    print("Aucune annotation utilisateur trouvée, aucune mise à jour nécessaire.")
