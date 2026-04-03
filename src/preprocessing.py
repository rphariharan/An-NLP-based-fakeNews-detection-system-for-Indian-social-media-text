import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocess the input text using standard NLP techniques.
    
    Steps:
    1. Convert to lowercase
    2. Clean noisy social media text (urls, mentions, special characters)
    3. Remove punctuation
    4. Tokenization
    5. Stopword removal
    6. Lemmatization
    """
    if not isinstance(text, str):
        return ""

    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Clean noisy social media text
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions (@username)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (can keep the word but remove the # symbol)
    text = re.sub(r'#', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 4. Tokenization
    # Using word_tokenize for better tokenization instead of simple split
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download('punkt_tab')
        tokens = word_tokenize(text)
    
    # 5. Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 6. Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Re-join tokens into a single string
    cleaned_text = ' '.join(lemmatized_tokens)
    
    return cleaned_text

if __name__ == "__main__":
    # Example before and after preprocessing
    sample_text = "Check out this NEW article at https://example.com!!! So many GRT insights. #FakeNews 😊 @user123"
    print("Original Text:", sample_text)
    print("Processed Text:", preprocess_text(sample_text))
