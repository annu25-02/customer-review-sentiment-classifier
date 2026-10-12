"""Train, evaluate and save the sentiment model."""
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from preprocess import clean_text

DATA_PATH = "data/reviews.csv"
MODEL_PATH = "models/sentiment_model.joblib"

# 1. Load data
df = pd.read_csv(DATA_PATH).dropna()
df["clean_review"] = df["review"].apply(clean_text)
print(f"Loaded {len(df)} reviews\n")
print(df["sentiment"].value_counts(), "\n")

# 2. Chart: class distribution
plt.figure(figsize=(5, 4))
ax = sns.countplot(data=df, x="sentiment", hue="sentiment", palette=["#2ca02c", "#d62728"],
                   order=["Positive", "Negative"], legend=False)
ax.set_title("Positive vs Negative Reviews")
for c in ax.containers:
    ax.bar_label(c)
plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=150)
plt.close()

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)

# 4. Model: TF-IDF (unigrams + bigrams so 'not good' is captured) -> Naive Bayes
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
    MultinomialNB(alpha=0.5),
)
model.fit(X_train, y_train)

# 5. Evaluate
preds = model.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, preds), 4))
cv = cross_val_score(model, df["clean_review"], df["sentiment"], cv=5)
print(f"5-fold CV accuracy: {cv.mean():.4f} (+/- {cv.std():.4f})\n")
print(classification_report(y_test, preds))

labels = ["Positive", "Negative"]
cm = confusion_matrix(y_test, preds, labels=labels)
ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# 6. Top words per class
vec, nb = model.named_steps["tfidfvectorizer"], model.named_steps["multinomialnb"]
names = vec.get_feature_names_out()
classes = list(nb.classes_)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, cls, color in zip(axes, ["Positive", "Negative"], ["#2ca02c", "#d62728"]):
    i = classes.index(cls)
    other = 1 - i
    score = nb.feature_log_prob_[i] - nb.feature_log_prob_[other]
    top = score.argsort()[-10:]
    ax.barh(names[top], score[top], color=color)
    ax.set_title(f"Words most indicating {cls}")
plt.tight_layout()
plt.savefig("outputs/top_words.png", dpi=150)
plt.close()

# 7. Save
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}; charts saved in outputs/")
