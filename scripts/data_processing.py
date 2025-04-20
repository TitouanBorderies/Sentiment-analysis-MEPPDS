from utils.utils import get_parquet_from_path


def load_data():
    """Load, clean and label train and validation data from parquet files."""
    train_path = "twitter_training.parquet"
    val_path = "twitter_validation.parquet"

    df_train = get_parquet_from_path(train_path)
    df_validation = get_parquet_from_path(val_path)

    columns = ["tweet_id", "entity", "sentiment", "text"]
    df_train.columns = columns
    df_validation.columns = columns

    # Clean train data
    df_train = df_train.dropna(subset=["sentiment", "text"])
    df_train["sentiment"] = df_train["sentiment"].str.strip().str.lower()
    df_train = df_train[df_train["sentiment"].isin(["negative", "neutral", "positive"])]
    df_train["label"] = df_train["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2}).astype(int)

    # Clean validation data
    df_validation = df_validation.dropna(subset=["sentiment", "text"])
    df_validation["sentiment"] = df_validation["sentiment"].str.strip().str.lower()
    df_validation = df_validation[df_validation["sentiment"].isin(["negative", "neutral", "positive"])]
    df_validation["label"] = df_validation["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2}).astype(int)

    return df_train, df_validation
