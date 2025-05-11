import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from typing import List

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def split_text_into_sentences(text: str) -> List[str]:
        """Divides the content of the input document in the original sentences"""

        sentences = sent_tokenize(text)
        return sentences

def tokenize(text:str) -> list[str]:
    """Tokenize text into a list of words/tokens."""
    return nltk.word_tokenize(text)

def clean_tokens(tokens: list[str]) -> list[str]:
    """Lowercase, remove punctuation and filter empty strings"""
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in string.punctuation]
    return [re.sub(r'^\W+|\W+$', '', token) for token in tokens if re.sub(r'^\W+|\W+$', '', token) != '']

def remove_stopwords(tokens: list[str]) -> list[str]:
    """Ignore common english stopwords"""
    return [token for token in tokens if token.lower() not in stop_words]

def get_wordnet_pos(treebank_tag: str) -> str:
    """Convert Treebank POS tags to WordNet POS tags for accurate lemmatization"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize(tokens: list[str]) -> list[str]:
    """Lematize tokens using english lemmatizer"""
    tagged_tokens = pos_tag(tokens)
    return [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged_tokens]

def preprocess(text: str) -> list[tuple[str, list[str]]]:
    """Full preprocessing"""
    processed = []

    sentences = split_text_into_sentences(text)
    for sentence in sentences:
        tokens = tokenize(sentence)
        tokens = clean_tokens(tokens)
        tokens = remove_stopwords(tokens)
        tokens = lemmatize(tokens)
        processed.append((sentence, tokens))

    return processed