import os
from transformers import TrainingArguments
from classes.trainer import CustomTrainer
from classes.architectures import CustomSentimentClassifier, tokenize_function
from scripts.data_processing import load_data
from classes.TweetDataset import TweetDataset
import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load training and validation datasets
df_train, df_validation = load_data()

# For debugging / quick testing
df_train = df_train.head(10).copy()
df_validation = df_validation.head(10).copy()

# Tokenize datasets
train_encodings = tokenize_function(df_train)
val_encodings = tokenize_function(df_validation)

train_dataset = TweetDataset(train_encodings, df_train["label"].tolist())
val_dataset = TweetDataset(val_encodings, df_validation["label"].tolist())

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=5,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=1,
    disable_tqdm=False,
    save_steps=10,
    save_total_limit=2,
    report_to="none",  # Disable default logging (e.g., to Hugging Face)
)

# Initialize model and trainer
model = CustomSentimentClassifier()
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# MLflow setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("bert-sentiment")

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("num_train_epochs", training_args.num_train_epochs)
    mlflow.log_param("train_batch_size", training_args.per_device_train_batch_size)
    mlflow.log_param("eval_batch_size", training_args.per_device_eval_batch_size)

    # Train model
    trainer.train()

    # Evaluate and log metrics
    eval_result = trainer.evaluate()
    for metric, value in eval_result.items():
        mlflow.log_metric(metric, value)

    # Save and log model
    model_path = os.path.join(training_args.output_dir, "final_model")
    trainer.save_model(model_path)
    mlflow.log_artifacts(model_path, artifact_path="model")
