import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tokenize(text:str) -> list[str]:
    """Tokenize text into a list of words/tokens."""
    return nltk.word_tokenize(text)

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

def preprocess(text: str) -> list[str]:
    """Full preprocessing"""
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return tokens