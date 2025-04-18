import os
from transformers import Trainer
import torch

class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Sauvegarder uniquement les poids du classificateur
        classifier_state_dict = self.model.classifier.state_dict()
        torch.save(classifier_state_dict, os.path.join(output_dir, "classifier_weights.bin"))

        # Sauvegarder le modèle TinyBERT (optionnel)
        self.model.tinybert.save_pretrained(output_dir)


