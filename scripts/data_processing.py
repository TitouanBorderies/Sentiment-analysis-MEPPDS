from utils.utils import get_parquet_from_path


def load_data():
    CHEMIN_FICHIER_TRAINING = "twitter_training.parquet"
    CHEMIN_FICHIER_VALIDATION = "twitter_validation.parquet"
    df_train = get_parquet_from_path(CHEMIN_FICHIER_TRAINING)
    df_validation = get_parquet_from_path(CHEMIN_FICHIER_VALIDATION)
    # Define column names
    column_names = ["tweet_id", "entity", "sentiment", "text"]
    df_train.columns = column_names
    df_validation.columns = column_names
    # Load and clean training data
    df_train = df_train.dropna(
        subset=["sentiment", "text"]
    )  # Drop rows with missing data
    df_train["sentiment"] = df_train["sentiment"].str.strip().str.lower()
    df_train = df_train[df_train["sentiment"].isin(["negative", "neutral", "positive"])]
    df_train["label"] = (
        df_train["sentiment"]
        .map({"negative": 0, "neutral": 1, "positive": 2})
        .astype(int)
    )

    # Load and clean validation data
    df_validation = df_validation.dropna(subset=["sentiment", "text"])
    df_validation["sentiment"] = df_validation["sentiment"].str.strip().str.lower()
    df_validation = df_validation[
        df_validation["sentiment"].isin(["negative", "neutral", "positive"])
    ]
    df_validation["label"] = (
        df_validation["sentiment"]
        .map({"negative": 0, "neutral": 1, "positive": 2})
        .astype(int)
    )
    return df_train, df_validation
