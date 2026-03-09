import pandas as pd
def sentiment_score(row):
    
    if row["label"] == "positive":
        return row["score"]
    
    elif row["label"] == "negative":
        return -row["score"]    
    else:
        return 0


def build_sentiment_index(df):

    df["date"] = pd.to_datetime(df["date"])

    df["sentiment_score"] = df.apply(sentiment_score, axis=1)

    sentiment_index = df.groupby("date")["sentiment_score"].mean()

    return sentiment_index