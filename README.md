# Customer Review Sentiment Classification (NLP + Naive Bayes)

Reads a customer review and predicts **Positive** or **Negative**.

**Pipeline:** Review → Text cleaning → TF-IDF (unigrams + bigrams) → Multinomial Naive Bayes → Prediction

## Project structure
```
generate_dataset.py   creates demo data/reviews.csv
preprocess.py         text cleaning (shared by training & prediction)
train.py              trains, evaluates, saves model + charts
predict.py            command-line predictions
app.py                Streamlit web app
data/  models/  outputs/
```

## Run it
```bash
pip install -r requirements.txt
python generate_dataset.py     # (skip if you use your own dataset)
python train.py                # prints metrics, saves model + charts
python predict.py "I love this phone"
streamlit run app.py           # web app
```

## Using a real dataset (recommended)
The bundled data is **synthetic and template-based**, so accuracy is unrealistically high.
For real results, download the [IMDb reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
or a product-review dataset, convert it to a CSV with columns `review,sentiment`
(values `Positive`/`Negative`), and save it as `data/reviews.csv`. Then re-run `python train.py`.

## Design choices
- **Negation kept:** stop-words like "not" are NOT removed, so "not good" stays negative.
- **Bigrams:** `ngram_range=(1,2)` lets the model learn phrases like "not good".
- **Evaluation:** accuracy, 5-fold cross-validation, precision/recall/F1, confusion matrix.

## Resume summary
Developed a sentiment classification system using Python and NLP to categorize customer
reviews as positive or negative. Used Pandas for data handling, TF-IDF for feature extraction,
and Multinomial Naive Bayes for training; evaluated with accuracy, precision, recall, F1 and a
confusion matrix; built a Streamlit app for single and bulk-CSV predictions.

## Ideas to extend
Compare with Logistic Regression / Linear SVM, add a Neutral class, tune hyperparameters with
GridSearchCV, deploy on Streamlit Community Cloud.
