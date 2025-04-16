import pandas as pd

def load_data():
    column_names = ['tweet_id', 'entity', 'sentiment', 'text']

    df_train = pd.read_csv("data/twitter_training.csv", header=None, names=column_names)
    df_validation = pd.read_csv("data/twitter_validation.csv", header=None, names=column_names)

    def preprocess(df):
        df = df.dropna(subset=['sentiment', 'text']).copy()
        df = df[df['sentiment'].isin(['negative', 'neutral', 'positive'])].copy()
        df.loc[:, 'sentiment'] = df['sentiment'].str.strip().str.lower()
        df.loc[:, 'label'] = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2}).astype(int)
        return df


    df_train = preprocess(df_train)
    df_validation = preprocess(df_validation)

    return df_train, df_validation
