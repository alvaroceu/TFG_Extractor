import nltk
from nltk.corpus import stopwords
import spacy

nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

def tokenize(text:str) -> list[str]:
    """Tokenize text into a list of words/tokens."""
    return nltk.word_tokenize(text)

def remove_stopwords(tokens: list[str]) -> list[str]:
    """Ignore common english stopwords"""
    return [token for token in tokens if token.lower() not in stop_words]

def lemmatize(tokens: list[str]) -> list[str]:
    """Lematize tokens using spaCy"""
    text = nlp(' '.join(tokens))
    return [token.lemma_ for token in text]

def preprocess(text: str) -> list[str]:
    """Full preprocessing"""
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return tokens