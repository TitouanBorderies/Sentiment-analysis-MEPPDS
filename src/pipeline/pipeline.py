def tokenize_data(df, tokenizer, max_length=128):
    return tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length
    )
