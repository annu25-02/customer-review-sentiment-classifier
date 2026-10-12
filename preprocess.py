"""Text cleaning used by both training and prediction (so they always match)."""
import re


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs/punctuation/digits, collapse whitespace.

    We deliberately KEEP words like 'not' and 'no' - removing them would turn
    'not good' into 'good' and flip the sentiment.
    """
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)          # HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)  # URLs
    text = re.sub(r"[^a-z\s']", " ", text)       # punctuation & digits
    text = re.sub(r"\s+", " ", text).strip()
    return text
