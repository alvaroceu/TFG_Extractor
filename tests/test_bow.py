import unittest
from traditional_bow.bow_extractor import *

class TestBoW(unittest.TestCase):

    def test_extract_match(self):
        extractor = BoWExtractor()
        text = "The capital of France is Paris. It is known for the Eiffel Tower."
        questions = """capital: What is the capital of France?
        dogs: Do you like dogs?"""
  
        result = extractor.extract(text, questions)
        assert result["capital"] == "The capital of France is Paris."
        assert result["dogs"] == "A possible valid answer wasn't found"