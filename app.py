import streamlit as st
import pickle
import os
import urllib.parse

from src.pipeline import final_prediction, get_evidence_links

# Set page configuration
st.set_page_config(
    page_title="Fact Verification System",
    page_icon="🔍",
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
        background-color: #1976D2;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1565C0;
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
    st.title("🔍 Fact Verification System")
    st.markdown("### Real-time Claim Verification Engine")
    st.markdown("---")
    
    # 1. Load Pre-trained Model and Vectorizer
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.error("Model files not found! Please make sure to run `python train_and_save.py` first to generate models in `saved_models/`.")
        return
        
    st.write("Enter an article excerpt, news, or claim below to verify its truthfulness.")
    
    # 2. Text input area
    news_text = st.text_area("Enter News or Claim", height=150, placeholder="E.g., The Eiffel tower is located in London.")
    
    # 3. Verify Now Button
    if st.button("Verify Now"):
        if not news_text.strip():
            st.warning("Please enter some text to verify.")
        else:
            with st.spinner("Querying knowledge base..."):
                # Real-time Prediction Module incorporating Semantic Fact Check
                final_verdict, sim_score, evidence, provenance = final_prediction(news_text, model, vectorizer)
                
                # Real-time Evidence Links fallback
                evidence_link = get_evidence_links(news_text)
                
                st.subheader("Verification Result")
                
                # UI Display: Prediction Verdict & Logic Mapping
                if final_verdict.upper() == 'FAKE':
                    st.markdown(f'<div class="fake-alert"><strong>❌ FINAL VERDICT: FAKE</strong><br><em>Result based on {provenance}</em></div>', unsafe_allow_html=True)
                elif final_verdict.upper() == 'REAL':
                    st.markdown(f'<div class="real-alert"><strong>✅ FINAL VERDICT: REAL</strong><br><em>Result based on {provenance}</em></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="color: #E65100; background-color: #FFF3E0; padding: 1rem; border-radius: 5px; border-left: 5px solid #F57C00; margin-top: 1rem;"><strong>⚠️ FINAL VERDICT: VERIFY</strong><br><em>Result based on {provenance}</em></div>', unsafe_allow_html=True)
                
                # Format semantic Confidence
                if provenance == "Semantic Verification" and isinstance(sim_score, float):
                    st.markdown(f"**Semantic Similarity Score:** {sim_score:.2f}")
                elif provenance != "Semantic Verification":
                    st.markdown(f"**ML Baseline Evaluator Warning** (Semantic logic bypassed)")
                
                # Specific explicit container for the Actual fact fetched 
                if evidence:
                    st.info(f"**Retrieved Evidence Text:**\n\n{evidence}")
                else:
                    st.warning("Could not fetch a definitive fact for this claim directly from Wikipedia. Please run independent searches.")
                    
                # UI Display: Real-time Google Evidence Link fallback
                st.markdown(f'''
                <div class="evidence-link">
                    <strong>🔍 Check alternative Real-Time Evidence:</strong><br>
                    <a href="{evidence_link}" target="_blank">Search Google for related facts based on keywords</a>
                </div>
                ''', unsafe_allow_html=True)
                    
    # Optional sidebar for info
    st.sidebar.title("About the System")
    st.sidebar.info("""
    Real-Time Fact Verification Pipeline.
    
    **Features:**
    - Priority Rules-Based Fallback logic natively defending standard truth queries
    - Live Wikipedia queries for dynamic Fact Checking
    - Baseline ML Predictions mapping unknown inferences via SVM `pickle`
    - Google Fact-check hyperlink queries
    """)

if __name__ == "__main__":
    main()
