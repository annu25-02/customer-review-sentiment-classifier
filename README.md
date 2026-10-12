# Customer Review Sentiment Classifier

A machine learning and Natural Language Processing (NLP) project that classifies customer reviews into **Positive** and **Negative** sentiments using TF-IDF vectorization and a Multinomial Naive Bayes classifier.

The project includes an interactive Streamlit web application for analyzing individual reviews and processing multiple reviews from a CSV file.

## Features

- Single customer review sentiment prediction
- Bulk sentiment analysis using CSV uploads
- Positive and Negative sentiment classification
- Prediction confidence scores and probability charts
- Sentiment distribution visualization for bulk predictions
- Downloadable CSV files containing prediction results
- Synthetic demo dataset generation for experimentation
- Command-line sentiment prediction

## Technologies Used

- Python
- Natural Language Processing (NLP)
- Machine Learning
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## Project Structure

```text
customer-review-sentiment-classifier/
├── app.py
├── predict.py
├── generate_dataset.py
├── preprocess.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── reviews.csv
└── models/
    └── sentiment_model.joblib
```

Note: `preprocess.py`, `train.py`, `requirements.txt`, the dataset, and the trained model are expected project files. Add them only when you have created or obtained the corresponding files.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/customer-review-sentiment-classifier.git
cd customer-review-sentiment-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset and model

Generate the synthetic demo dataset:

```bash
python generate_dataset.py
```

Run the model training script to create the trained model file before launching the application.

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

### 6. Run command-line predictions

```bash
python predict.py "The product is amazing and works perfectly"
```

## Dataset

The supplied dataset-generation script creates synthetic customer reviews using predefined positive and negative templates. It is intended for learning and demonstrating the ML pipeline.

**Important:** Synthetic template-based data can produce unrealistically high evaluation scores. Real-world performance should be evaluated using an independent, representative dataset.

## Applications

- Customer feedback analysis
- Product review classification
- Basic sentiment analysis demonstrations
- Bulk review processing

## Future Improvements

- Support for neutral sentiment
- Training and evaluation on real-world review datasets
- Model performance metrics and confusion matrix
- Improved handling of negation, sarcasm, and mixed sentiment
- Deployment as a publicly accessible web application

## Author

Add your name and GitHub profile link here.

## Disclaimer

This is an educational machine learning project. Predictions are estimates and should not be treated as definitive interpretations of customer opinions.
