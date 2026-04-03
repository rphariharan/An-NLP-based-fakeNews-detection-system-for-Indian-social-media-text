# Fake News Detection in Indian Social Media Text

This project provides a complete end-to-end pipeline for fake news detection based on NLP techniques and Machine Learning models.

## User Review Required
Please review the proposed project structure and the libraries being used. Once approved, I will proceed to implement the modules in the `c:\Users\rphar\Downloads\miniproject` directory.

## Proposed Changes

### Configuration and Setup
#### [NEW] [requirements.txt](file:///c:/Users/rphar/Downloads/miniproject/requirements.txt)
- Will contain necessary libraries: `pandas`, `nltk`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`.

### Data
#### [NEW] [dummy_dataset.csv](file:///c:/Users/rphar/Downloads/miniproject/data/dummy_dataset.csv)
- A small sample dataset for testing the pipeline out-of-the-box (with columns "text" and "label").

### Source Code (`src/` module)
#### [NEW] [preprocessing.py](file:///c:/Users/rphar/Downloads/miniproject/src/preprocessing.py)
- `preprocess_text(text)` function using NLTK: lowercase, punctuation removal, tokenization, stopwords, lemmatization.
#### [NEW] [features.py](file:///c:/Users/rphar/Downloads/miniproject/src/features.py)
- TF-IDF vectorization handling unigram and bigram features.
#### [NEW] [models.py](file:///c:/Users/rphar/Downloads/miniproject/src/models.py)
- Training Naive Bayes and SVM models.
- Model evaluation (Accuracy, Precision, Recall, F1-score, Confusion Matrix).
#### [NEW] [pipeline.py](file:///c:/Users/rphar/Downloads/miniproject/src/pipeline.py)
- `predict_news(text)` function combining preprocessing, feature extraction, and prediction.

### Entry Points
#### [NEW] [main.py](file:///c:/Users/rphar/Downloads/miniproject/main.py)
- CLI script to orchestrate dataset loading, training models, evaluation, and displaying the comparative analysis table as requested.
#### [NEW] [app.py](file:///c:/Users/rphar/Downloads/miniproject/app.py)
- Streamlit application providing a simple user interface for real-time fake news prediction.

## Verification Plan

### Automated/CLI Tests
- Run `python main.py` to ensure the full pipeline runs without errors, training models and producing the required evaluation metrics and comparative table.

### Manual Verification
- Run `streamlit run app.py` to launch the web interface.
- Paste test text (e.g., "Government giving ₹10000 to every citizen apply now") and verify the system predicts Fake/Real along with a confidence score.
