import urllib.parse
import wikipedia
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

KNOWN_FACTS = {
    "president of india": ("Droupadi Murmu", "President of India"),
    "prime minister of india": ("Narendra Modi", "Prime Minister of India"),
    "capital of india": ("New Delhi", "New Delhi"),
    "currency of india": ("Indian Rupee", "Indian rupee")
}

def fact_check(text):
    """
    Fact verification module using rule-based matching overlaid with Wikipedia API.
    Detects specific topics, compares with known facts, and fetches the real summary.
    """
    text_lower = text.lower()
    
    matched_topic = None
    correct_value = None
    wiki_search_term = None
    
    # 1. Rule-based topic detection
    for topic, (val, wiki_term) in KNOWN_FACTS.items():
        if topic in text_lower:
            matched_topic = topic
            correct_value = val
            wiki_search_term = wiki_term
            break
            
    if matched_topic:
        # 2. Compare Claimed Value
        # Uses case-insensitive substring matching to detect if the correct answer is mentioned
        if correct_value.lower() in text_lower:
            label = "REAL"
        else:
            label = "FAKE"
            
        # 3. Fetch exact summary for the correct topic
        try:
            summary = wikipedia.summary(wiki_search_term, sentences=1, auto_suggest=False)
        except Exception:
            # Fallback if Wikipedia fails securely
            summary = f"The actual {matched_topic} is {correct_value}."
            
        return label, summary
        
    # No known topic found, fallback to VERIFY mode
    return "VERIFY", ""

def final_prediction(text, model, vectorizer):
    """
    Prediction flow integrating both ML baseline and Wikipedia Fact Check.
    """
    # 1. Run standard ML prediction
    ml_pred, confidence = predict_news(text, model, vectorizer)
    
    # 2. Run fact verification
    fact_label, actual_fact = fact_check(text)
    
    # 3. Combine results based on logical fallback
    if fact_label == "FAKE":
        final_label = "Fake"
    elif fact_label == "REAL":
        final_label = "Real"
    else:
        final_label = ml_pred
        
    return final_label, confidence, actual_fact

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
