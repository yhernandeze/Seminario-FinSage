import pandas as pd
from preprocessing import clean_text
from finbert_model import predict_sentiment
from sentiment_index import build_sentiment_index

def run_pipeline():
    df = pd.read_csv("data/raw/news.csv")
    df["clean_headline"] = df["headline"].apply(clean_text)
    df["sentiment"] = df["clean_headline"].apply(predict_sentiment)
    sentiment_index = build_sentiment_index(df)
    sentiment_index.to_csv("data/processed/sentiment_index.csv")
    print("Pipeline completado")
if __name__ == "__main__":
    run_pipeline()