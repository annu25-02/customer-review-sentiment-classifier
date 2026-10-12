"""Streamlit web app.  Run:  streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st
from preprocess import clean_text

st.set_page_config(page_title="Review Sentiment Classifier", page_icon="💬")


@st.cache_resource
def load_model():
    return joblib.load("models/sentiment_model.joblib")


model = load_model()

st.title("💬 Customer Review Sentiment Classifier")
st.caption("TF-IDF + Multinomial Naive Bayes")

tab1, tab2 = st.tabs(["Single review", "Bulk CSV"])

with tab1:
    review = st.text_area("Enter a customer review", height=120,
                          placeholder="The product is amazing and I love it!")
    if st.button("Predict", type="primary"):
        if review.strip():
            cleaned = clean_text(review)
            label = model.predict([cleaned])[0]
            probs = dict(zip(model.classes_, model.predict_proba([cleaned])[0]))
            (st.success if label == "Positive" else st.error)(
                f"Prediction: **{label}** ({probs[label]:.0%} confidence)")
            st.bar_chart(pd.Series(probs, name="Probability"))
        else:
            st.warning("Please type a review first.")

with tab2:
    st.write("Upload a CSV with a column named **review**.")
    file = st.file_uploader("CSV file", type="csv")
    if file:
        data = pd.read_csv(file)
        if "review" not in data.columns:
            st.error("CSV must contain a 'review' column.")
        else:
            data["prediction"] = model.predict(data["review"].apply(clean_text))
            st.dataframe(data)
            counts = data["prediction"].value_counts()
            st.write("Share of reviews:")
            st.bar_chart(counts / counts.sum() * 100)
            st.download_button("Download results", data.to_csv(index=False),
                               "predictions.csv", "text/csv")
