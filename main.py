import os
from bluesky import initialize_client, get_last_message
from architectures import CustomSentimentClassifier
from dotenv import load_dotenv

load_dotenv()
client = initialize_client()
dernier_message = get_last_message(client)

path_model = os.environ.get("MODEL_PATH", "")
model = CustomSentimentClassifier.from_pretrained(path_model)

print("Dernier message de @lemonde.fr :")
print(dernier_message)
print("Son sentiment est " + model.infer_sentiment(dernier_message))
