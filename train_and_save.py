import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_text
from src.features import extract_features
from src.models import train_svm, train_naive_bayes

def main():
    print("Loading dataset...")
    # 1. Load dataset
    dataset_path = 'data/dummy_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return
        
    df = pd.read_csv(dataset_path)
    
    print("Preprocessing text...")
    # 2. Apply preprocessing
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Simple split
    X = df['cleaned_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Extracting TF-IDF features...")
    # 3. Apply TF-IDF Feature Extraction
    # We fit the vectorizer on the training data using extract_features logic
    # using bigrams for better context capture
    X_train_features, vectorizer = extract_features(X_train, use_bigrams=True)
    
    print("Training SVM model...")
    # 4. Train Model (SVM works best due to linear kernel + probability calibration)
    svm_model = train_svm(X_train_features, y_train)
    
    # Create saved_models directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    print("Saving models using pickle...")
    # 5. Save the trained model and vectorizer
    with open('saved_models/svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
        
    with open('saved_models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Success: Model and vectorizer saved in 'saved_models/' directory.")

if __name__ == "__main__":
    main()
