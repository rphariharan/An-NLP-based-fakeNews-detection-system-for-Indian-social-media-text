import urllib.parse
from .preprocessing import preprocess_text

def predict_news(text, model, vectorizer):
    """
    Real-Time Prediction Module.
    
    This function takes a trained model and vectorizer along with raw text.
    It preprocesses the text, converts it to TF-IDF features, and predicts Fake or Real.
    
    Parameters:
    - text: str, the raw news article text
    - model: trained sklearn model (e.g., Naive Bayes or SVM)
    - vectorizer: fitted TfidfVectorizer
    
    Returns:
    - prediction: str, "Fake" or "Real"
    - confidence: float, percentage confidence score
    """
    # 1. Preprocess the text
    cleaned_text = preprocess_text(text)
    
    if not cleaned_text:
        return "Unknown", 0.0
    
    # 2. Convert using TF-IDF vectorizer
    features = vectorizer.transform([cleaned_text])
    
    # 3. Use trained model for prediction
    pred_label = model.predict(features)[0]
    
    # 4. Get confidence score as percentage
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(features)[0]
        # Max probability out of the classes
        confidence = max(probas) * 100.0
    elif hasattr(model, "decision_function"):
        # Normalizing logic purely for visualization without probas
        decision = model.decision_function(features)[0]
        confidence = min(max(abs(decision) * 50.0, 50.0), 99.9)
        
    return pred_label, confidence

def get_evidence_links(text):
    """
    Extracts keywords from the input text and generates a Google search URL 
    to provide real-time evidence support.
    
    Parameters:
    - text: str, original raw input text (will perform basic tokenization/cleaning here to pull keywords)
    
    Returns:
    - string URL pointing to Google search
    """
    cleaned_text = preprocess_text(text)
    
    if not cleaned_text:
        return "https://www.google.com"
        
    # Extract top keywords (take first 5 meaningful tokens)
    keywords = cleaned_text.split()[:5]
    query = " ".join(keywords)
    
    # Encode for URL appending
    encoded_query = urllib.parse.quote_plus(query)
    
    google_url = f"https://www.google.com/search?q={encoded_query}"
    return google_url
