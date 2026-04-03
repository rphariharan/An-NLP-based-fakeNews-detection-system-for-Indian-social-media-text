import streamlit as st
import pickle
import os
import urllib.parse

from src.pipeline import predict_news, get_evidence_links

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .fake-alert {
        color: #D32F2F;
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #D32F2F;
        margin-top: 1rem;
    }
    .real-alert {
        color: #388E3C;
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #388E3C;
        margin-top: 1rem;
    }
    .evidence-link {
        margin-top: 10px;
        padding: 10px;
        background-color: #f1f8ff;
        border-left: 5px solid #0366d6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """
    Load pre-trained classification model and vectorizer from pickle files.
    This ensures we do NOT retrain the model during runtime.
    """
    model_path = 'saved_models/svm_model.pkl'
    vec_path = 'saved_models/tfidf_vectorizer.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

def main():
    st.title("📰 Fake News Detection System")
    st.markdown("### Real-time Prediction Engine")
    st.markdown("---")
    
    # 1. Load Pre-trained Model and Vectorizer
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.error("Model files not found! Please make sure to run `python train_and_save.py` first to generate models in `saved_models/`.")
        return
        
    st.write("Enter the news text or social media forward below to get instant real-time prediction.")
    
    # 2. Text input area
    news_text = st.text_area("Enter News Text", height=150, placeholder="E.g., Government giving ₹10000 to every citizen apply now")
    
    # 3. Check News Button
    if st.button("Check News"):
        if not news_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing real-time..."):
                # Real-time Prediction Module
                prediction, confidence = predict_news(news_text, model, vectorizer)
                
                # Real-time Evidence Links
                evidence_link = get_evidence_links(news_text)
                
                # Format confidence as percentage
                if isinstance(confidence, float):
                    conf_str = f"Confidence: {confidence:.2f}%"
                else:
                    conf_str = "High Confidence"
                
                st.subheader("Real-Time Analysis Result")
                
                # UI Display: Prediction & Confidence
                if prediction.lower() == 'fake':
                    st.markdown(f'<div class="fake-alert"><strong>🚨 THIS APPEARS TO BE FAKE NEWS</strong><br>{conf_str}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="real-alert"><strong>✅ THIS APPEARS TO BE REAL NEWS</strong><br>{conf_str}</div>', unsafe_allow_html=True)
                
                # UI Display: Real-time Evidence Link
                st.markdown(f'''
                <div class="evidence-link">
                    <strong>🔍 Check Real-Time Evidence:</strong><br>
                    <a href="{evidence_link}" target="_blank">Search Google for related facts based on keywords</a>
                </div>
                ''', unsafe_allow_html=True)
                    
    # Optional sidebar for info
    st.sidebar.title("About the System")
    st.sidebar.info("""
    Real-Time Fake News Detection Pipeline.
    
    **Features:**
    - Loads pre-trained models via `pickle`
    - No runtime retraining (Fast Response)
    - Keyword extraction & Fact-check Google Search Support
    
    *Powered by SVM & Streamlit.*
    """)

if __name__ == "__main__":
    main()
