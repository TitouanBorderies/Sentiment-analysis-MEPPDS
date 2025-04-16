# main.py
import os
import argparse
from dotenv import load_dotenv
from loguru import logger
from joblib import dump

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import mlflow

from src.data.data import load_data
from src.models.models import TweetDataset
from src.pipeline.pipeline import tokenize_data

# ENVIRONMENT CONFIGURATION ---------------------------
logger.add("recording.log", rotation="500 MB")
load_dotenv()

parser = argparse.ArgumentParser(description="BERT for Sequence Classification")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--experiment_name", type=str, default="bert_seq_classification", help="MLFlow experiment name")
args = parser.parse_args()

# Load environment variables
model_name = os.getenv("MODEL_NAME", "bert-base-uncased")
data_train_path = os.getenv("train_path", "data/processed/train.parquet")
data_validation_path = os.getenv("validation_path", "data/processed/validation.parquet")
logger.info(f"Using model {model_name}")

# LOAD AND PREPROCESS DATA ----------------------------
df_train, df_validation = load_data()
logger.info(f"Train DataFrame shape: {df_train.shape}")
logger.info(f"Train DataFrame columns: {df_train.columns.tolist()}")
logger.info(f"Sample rows:\n{df_train.head()}")

# Tokenization
tokenizer = BertTokenizer.from_pretrained(model_name)
encodings_train = tokenize_data(df_train, tokenizer)
encodings_validation = tokenize_data(df_validation, tokenizer)

# Dataset creation
train_dataset = TweetDataset(encodings_train, df_train['label'].tolist())
validation_dataset = TweetDataset(encodings_validation, df_validation['label'].tolist())

# MODEL AND TRAINING SETUP ----------------------------
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=64,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

# TRAINING ----------------------------
trainer.train()

with open("bert_model.joblib", "wb") as f:
    dump(model, f)

# EVALUATION ----------------------------
metrics = trainer.evaluate()
logger.info(f"Evaluation metrics: {metrics}")

# LOGGING IN MLFLOW -----------------
mlflow_server = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(mlflow_server)
mlflow.set_experiment(args.experiment_name)

with mlflow.start_run():
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("model_name", model_name)
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)
    mlflow.pytorch.log_model(model, "model")
    logger.success("Training and evaluation logged successfully.")
