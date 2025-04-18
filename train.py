from transformers import TrainingArguments
from trainer import CustomTrainer
from architectures import CustomSentimentClassifier, tokenize_function
from data_processing import load_data
from TweetDataset import TweetDataset

df_train, df_validation = load_data()

df_train = df_train.head(10).copy()
df_validation = df_validation.head(10).copy()

train_encodings = tokenize_function(df_train)
val_encodings = tokenize_function(df_validation)

train_dataset = TweetDataset(train_encodings, df_train['label'].tolist())
val_dataset = TweetDataset(val_encodings, df_validation['label'].tolist())



training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=1,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=1,
    disable_tqdm=False,
    save_steps=10,
    save_total_limit=2,
)

model = CustomSentimentClassifier()
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
