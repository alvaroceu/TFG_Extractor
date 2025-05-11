import unittest
from traditional_bow.preprocessing import *
from traditional_bow.bow_extractor import *

class TestPreprocessing(unittest.TestCase):

    def test_tokenize(self):
        text = "Hello World!"
        tokens = tokenize(text)
        self.assertIn("Hello", tokens)
        self.assertIn("World", tokens)
    
    def test_remove_stopwords(self):
        tokens = ["this", "is", "a", "test"]
        no_stopwords = remove_stopwords(tokens)
        self.assertNotIn("a", no_stopwords)
        self.assertIn("test", no_stopwords)
    
    def test_lemmatize(self):
        tokens = ["running", "flies"]
        lemmatized = lemmatize(tokens)
        self.assertIn("run", lemmatized)
        self.assertIn("fly", lemmatized)
    
    def test_full_preprocess(self):
        text = "The quick brown fox jumps over the lazy dogs near the river, looking for something interesting to eat."
        lemmatized = preprocess(text)
        tokens = lemmatized[0][1]
        self.assertIn("fox", tokens)
        self.assertIn("jump", tokens)
        self.assertIn("look", tokens)
        self.assertIn("eat", tokens)
        self.assertNotIn("The", tokens)
        self.assertNotIn("for", tokens)
        self.assertNotIn("to", tokens)

class TestBoW(unittest.TestCase):

    def test_extract_match(self):
        extractor = BoWExtractor()
        text = "The capital of France is Paris. It is known for the Eiffel Tower."
        questions = """capital: What is the capital of France?
        dogs: Do you like dogs?"""
  
        result = extractor.extract(text, questions)
        assert result["capital"] == "The capital of France is Paris."
        assert result["dogs"] == "A possible valid answer wasn't found"