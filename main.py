import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Force utf-8 encoding for standard output to handle ₹ and other characters on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Import custom modules
from src.preprocessing import preprocess_text
from src.features import extract_features
from src.models import train_naive_bayes, train_svm, evaluate_model, compare_models
from src.pipeline import predict_news

def main():
    print("=" * 50)
    print("NLP-Based Comparative Study of Fake News Detection")
    print("=" * 50)
    
    # 1. Dataset Handling
    dataset_path = 'data/dummy_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please create it first.")
        return
        
    print("\n--- 1. Loading Dataset ---")
    df = pd.read_csv(dataset_path)
    
    print("Dataset Summary:")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nSample Data:")
    print(df.head(3))
    
    # 2. Clean/Preprocess Dataset (Can take a while on large datasets)
    print("\n--- 2. Preprocessing Text ---")
    # Apply preprocessing to all texts
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Split dataset (80/20)
    X = df['cleaned_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size:  {len(X_test)}")
    
    # 3. Feature Extraction
    print("\n--- 3. Extracting Features (TF-IDF) ---")
    # Using Unigram + Bigram
    print("Using Unigram + Bigram implementation...")
    
    # Fit vectorizer only on training data, transform both train and test
    # Or simply we can use extract_features for train, then transform test
    # We will modify extract_features slightly for train/test split usage here implicitly.
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    print(f"TF-IDF Matrix dimensions (Train): {X_train_features.shape}")
    
    # 4. Machine Learning Models
    print("\n--- 4. Training Models ---")
    
    print("Training Naive Bayes...")
    nb_model = train_naive_bayes(X_train_features, y_train)
    
    print("Training Support Vector Machine (SVM)...")
    svm_model = train_svm(X_train_features, y_train)
    
    # 5. Model Evaluation
    print("\n--- 5. Evaluating Models ---")
    metrics_list = []
    
    # Evaluate NB
    nb_metrics = evaluate_model(nb_model, X_test_features, y_test, model_name="Naive Bayes", plot_cm=False)
    metrics_list.append(nb_metrics)
    
    # Evaluate SVM
    svm_metrics = evaluate_model(svm_model, X_test_features, y_test, model_name="SVM", plot_cm=False)
    metrics_list.append(svm_metrics)
    
    # 6. Comparative Analysis
    print("\n--- 6. Comparative Analysis ---")
    compare_models(metrics_list)
    
    # 7. Real-Time Prediction Module (Test)
    print("\n--- 7. Real-Time Prediction Test ---")
    sample_text = "Government giving ₹10000 to every citizen apply now"
    print(f"Input Text: '{sample_text}'")
    
    # We choose SVM for the demo since it usually performs well
    prediction, confidence = predict_news(sample_text, svm_model, vectorizer)
    
    print(f"Prediction: {prediction}")
    if isinstance(confidence, float) and confidence <= 1.0:
        print(f"Confidence: {confidence*100:.2f}%")
    else:
        print(f"Wait/Distance score: {confidence:.4f}")

if __name__ == "__main__":
    main()
