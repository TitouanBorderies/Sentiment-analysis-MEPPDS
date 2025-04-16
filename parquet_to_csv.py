import pandas as pd

# Charger le fichier Parquet
df = pd.read_parquet("twitter_training.parquet", engine="pyarrow")

# Sauvegarder en format CSV
df.to_csv("twitter_training.csv", index=False)
