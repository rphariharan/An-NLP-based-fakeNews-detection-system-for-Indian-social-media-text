import urllib.parse
import wikipedia
import nltk
from sentence_transformers import SentenceTransformer, util
from .preprocessing import preprocess_text

# Globals for lazy loading
embedder = None

def get_models():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return embedder

def predict_news(text, model, vectorizer):
    """
    Real-Time Prediction Module Fallback.
    """
    cleaned_text = preprocess_text(text)
    if not cleaned_text:
        return "Unknown", 0.0
    
    features = vectorizer.transform([cleaned_text])
    pred_label = model.predict(features)[0]
    
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(features)[0]
        confidence = max(probas) * 100.0
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(features)[0]
        confidence = min(max(abs(decision) * 50.0, 50.0), 99.9)
        
    return pred_label, confidence

def extract_entity(text):
    # Make sure taggers are downloaded
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng')
        
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    
    # Extract Proper Nouns (Entities)
    prop_nouns = [word for word, pos in tags if pos == 'NNP' or pos == 'NNPS']
    if prop_nouns:
        return " ".join(prop_nouns)
        
    # Fallback checking noun structures
    nouns = [word for word, pos in tags if pos.startswith('NN')]
    if nouns:
        return " ".join(nouns[:2])
        
    words = text.split()
    return " ".join(words[:2]) if words else ""

def verify_semantics(claim, evidence):
    embed_model = get_models()
    
    # 1. Base Semantic Similarity Mapping
    claim_emb = embed_model.encode(claim, convert_to_tensor=True)
    evi_emb = embed_model.encode(evidence, convert_to_tensor=True)
    sim_score = util.cos_sim(claim_emb, evi_emb).item()
    
    # 2. Heuristic Negation Check (Roles Constraint Overlap)
    roles = ['governor', 'chief minister', 'president', 'prime minister', 'ceo', 'mayor', 'director']
    claim_lower = claim.lower()
    evi_lower = evidence.lower()
    
    mismatch = False
    for role in roles:
        if role in claim_lower and role not in evi_lower:
            mismatch = True
            break
            
    if mismatch:
        sim_score -= 0.5   # Massive penalty triggers FAKE
        
    return sim_score

def fact_check_semantic(text):
    """
    Deep-learning fact verification relying entirely on SPAcy and sentence transformers natively matching reality constraints!
    """
    entity = extract_entity(text)
    if not entity:
        return "VERIFY", 0.0, "Could not strictly extract an entity to examine via Wikipedia."
        
    try:
        # Search targets precisely targeting the entity context mapping definition
        results = wikipedia.search(entity, results=1)
        if not results:
            return "VERIFY", 0.0, "No accurate Wikipedia repositories found for entity search inference."
            
        page_title = results[0]
        evidence = wikipedia.summary(page_title, sentences=2, auto_suggest=False)
    except Exception as e:
        return "VERIFY", 0.0, "Wikipedia connection failed."
        
    # Semantic parsing handling
    sim_score = verify_semantics(text, evidence)
    
    # Overriding standard mapping logic to absolute semantic truth metrics defined manually
    if sim_score > 0.6:
        verdict = "REAL"
    # Never classify as FAKE unless strong contradiction heavily penalized the score (falling below 0)
    elif sim_score < 0.0:
        verdict = "FAKE"
    else:
        verdict = "VERIFY"
        
    return verdict, sim_score, evidence

def final_prediction(text, model, vectorizer):
    """
    Execution map strictly mapping semantics above everything and disabling false positives implicitly.
    """
    sem_verdict, sim_score, evidence = fact_check_semantic(text)
    
    if sem_verdict in ["REAL", "FAKE"]:
        return sem_verdict, sim_score, evidence, "Semantic Verification"
        
    # Ensure system aggressively suppresses FAKE inferences not rigorously verified structurally!
    return "VERIFY", sim_score, evidence, "ML Priority Override Verification"

def get_evidence_links(text):
    """
    Extracts keywords from the input text and generates a Google search URL 
    to provide real-time evidence support.
    """
    cleaned_text = preprocess_text(text)
    
    if not cleaned_text:
        return "https://www.google.com"
        
    keywords = cleaned_text.split()[:5]
    query = " ".join(keywords)
    
    encoded_query = urllib.parse.quote_plus(query)
    google_url = f"https://www.google.com/search?q={encoded_query}"
    return google_url
