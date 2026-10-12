"""Command-line prediction:  python predict.py "The product is amazing" """
import sys
import joblib
from preprocess import clean_text

model = joblib.load("models/sentiment_model.joblib")


def predict(review: str):
    cleaned = clean_text(review)
    label = model.predict([cleaned])[0]
    conf = model.predict_proba([cleaned]).max()
    return label, conf


if __name__ == "__main__":
    reviews = sys.argv[1:] or [
        "The product is amazing. The quality is excellent, and I love it!",
        "The product is terrible. It stopped working after two days.",
        "The service was not good at all",
    ]
    for r in reviews:
        label, conf = predict(r)
        print(f"{label:9s} ({conf:.0%})  <-  {r}")
