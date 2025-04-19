import torch
import os
from transformers import AutoModel, AutoTokenizer
from torch import nn


#### Tokenizer ####
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")


def tokenize_function(examples):
    return tokenizer(
        examples["text"].tolist(), padding="max_length", truncation=True, max_length=128
    )


#### Classifier ####
class CustomSentimentClassifier(nn.Module):
    def __init__(self, hidden_size=312, num_labels=3):
        super(CustomSentimentClassifier, self).__init__()
        self.n_neurons = 50
        self.tinybert = AutoModel.from_pretrained(
            "huawei-noah/TinyBERT_General_4L_312D"
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, self.n_neurons), nn.ReLU(),
            nn.Linear(self.n_neurons, self.n_neurons), nn.ReLU(), 
            nn.Linear(self.n_neurons, num_labels)
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.tinybert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[
            :, 0, :
        ]  # Utiliser le premier token [CLS]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

    def infer_sentiment(self, text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        outputs = self(**inputs)
        logits = outputs["logits"]
        predicted_class = torch.argmax(logits, dim=1).item()
        return_dict = {0: "négatif", 1: "neutre", 2: "Positif"}
        return return_dict[predicted_class]

    @classmethod
    def from_pretrained(cls, pretrained_model_dir, hidden_size=312, num_labels=3):
        model = cls(hidden_size=hidden_size, num_labels=num_labels)
        classifier_file = os.path.join(pretrained_model_dir, "classifier_weights.bin")
        classifier_state_dict = torch.load(
            classifier_file, map_location=torch.device("cpu")
        )
        model.classifier.load_state_dict(classifier_state_dict)
        return model
