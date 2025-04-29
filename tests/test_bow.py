import unittest
from traditional_bow.preprocessing import *

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
        text = "The quick brown fox jumps over the lazy dog near the river, looking for something interesting to eat."
        lemmatized = preprocess(text)
        self.assertIn("fox", lemmatized)
        self.assertIn("jump", lemmatized)
        self.assertIn("look", lemmatized)
        self.assertIn("eat", lemmatized)
        self.assertNotIn("The", lemmatized)
        self.assertNotIn("for", lemmatized)
        self.assertNotIn("to", lemmatized)