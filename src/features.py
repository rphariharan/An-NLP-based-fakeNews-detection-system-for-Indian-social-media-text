from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def extract_features(texts, use_bigrams=False, max_features=None):
    """
    Convert processed text into numerical features using TF-IDF vectorization.
    
    Parameters:
    - texts: Iterable of string, the corpus to vectorize
    - use_bigrams: Boolean, whether to include bigrams in addition to unigrams
    - max_features: int, maximum number of features to extract
    
    Returns:
    - tfidf_matrix: Sparse matrix of TF-IDF features
    - vectorizer: The fitted TfidfVectorizer instance
    """
    
    if use_bigrams:
        # ngram_range=(1, 2) means unigrams and bigrams
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    else:
        # Default is unigram only: ngram_range=(1, 1)
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=max_features)
        
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    return tfidf_matrix, vectorizer

if __name__ == "__main__":
    # Example to show TF-IDF matrix dimensions
    sample_corpus = [
        "fake news detection project",
        "this is real news",
        "machine learning works well for fake news"
    ]
    
    matrix, vec = extract_features(sample_corpus, use_bigrams=False)
    print("Unigram TF-IDF matrix dimensions:", matrix.shape)
    
    matrix_bigram, vec_bigram = extract_features(sample_corpus, use_bigrams=True)
    print("Unigram + Bigram TF-IDF matrix dimensions:", matrix_bigram.shape)
