from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = probs.detach().numpy()[0]
    result = {
        "positive": float(sentiment[0]),
        "negative": float(sentiment[1]),
        "neutral": float(sentiment[2])
    }

    return result