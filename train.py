import os
import argparse
from dotenv import load_dotenv
from loguru import logger
from joblib import dump

import pathlib
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import mlflow

# ENVIRONMENT CONFIGURATION ---------------------------

logger.add("recording.log", rotation="500 MB")
load_dotenv()

# Set up argument parsing for flexibility
parser = argparse.ArgumentParser(description="BERT for Sequence Classification")
parser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size for training"
)
parser.add_argument(
    "--epochs", type=int, default=3, help="Number of training epochs"
)
parser.add_argument(
    "--experiment_name", type=str, default="bert_seq_classification", help="MLFlow experiment name"
)
args = parser.parse_args()

# Load environment variables for flexibility
model_name = os.environ.get("MODEL_NAME", "bert-base-uncased")
data_train_path = os.environ.get("train_path", "data/processed/train.parquet")
data_test_path = os.environ.get("test_path", "data/processed/test.parquet")
logger.info(f"Using model {model_name}")

# IMPORT ET STRUCTURATION DONNEES --------------------------------

# Define column names
column_names = ['tweet_id', 'entity', 'sentiment', 'text']

# Load and clean the training data
df_train = pd.read_csv("/home/onyxia/work/Sentiment-analysis-MEPPDS/data/twitter_training.csv", header=None, names=column_names)
df_train = df_train.dropna(subset=['sentiment', 'text'])
df_train['sentiment'] = df_train['sentiment'].str.strip().str.lower()
df_train = df_train[df_train['sentiment'].isin(['negative', 'neutral', 'positive'])]
df_train['label'] = df_train['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).astype(int)

# Load and clean the validation data
df_validation = pd.read_csv("/home/onyxia/work/Sentiment-analysis-MEPPDS/data/twitter_validation.csv", header=None, names=column_names)
df_validation = df_validation.dropna(subset=['sentiment', 'text'])
df_validation['sentiment'] = df_validation['sentiment'].str.strip().str.lower()
df_validation = df_validation[df_validation['sentiment'].isin(['negative', 'neutral', 'positive'])]
df_validation['label'] = df_validation['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).astype(int)

# Tokenization
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_data(df, tokenizer, max_length=128):
    encodings = tokenizer(df['text'].tolist(), truncation=True, padding=True, max_length=max_length)
    return encodings

encodings_train = tokenize_data(df_train, tokenizer)
encodings_validation = tokenize_data(df_validation, tokenizer)

# Dataset class
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        } | {'labels': torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(encodings_train, df_train['label'].tolist())
validation_dataset = TweetDataset(encodings_validation, df_validation['label'].tolist())

# Define the model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=64,
    logging_dir="./logs",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

# TRAINING ----------------------------

# Train the model
trainer.train()

# Save the model
with open("bert_model.joblib", "wb") as f:
    dump(model, f)

# EVALUATION ----------------------------

# Evaluation could be done through Trainer's built-in evaluation
metrics = trainer.evaluate()

logger.info(f"Evaluation metrics: {metrics}")

# LOGGING IN MLFLOW -----------------

mlflow_server = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(mlflow_server)
mlflow.set_experiment(args.experiment_name)

# Log model and metrics in MLFlow
with mlflow.start_run():

    # Log parameters
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("model_name", model_name)

    # Log evaluation metrics
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)

    # Log model
    mlflow.pytorch.log_model(model, "model")

    logger.success("Training and evaluation logged successfully.")




