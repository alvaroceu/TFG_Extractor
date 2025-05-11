from core.extractor_base import ExtractorBase
from traditional_bow.preprocessing import preprocess
from traditional_bow.preprocessing import preprocess_questions
from typing import List

class BoWExtractor(ExtractorBase):

    def extract(self, text: str, questions: str):
        """Extract relevant information from text for each column."""
        
        results = {}

        preprocessed_sentences = preprocess(text)
        bags_of_words = preprocess_questions(questions)

        for column, bag_of_words in bags_of_words.items():
            best_score = 0
            best_answer = "A possible valid answer wasn't found"

            for sentence, sentence_tokens in preprocessed_sentences:

                score = self.similarity_score(sentence_tokens, bag_of_words)
                if score > best_score:
                    best_score = score
                    best_answer = sentence

            results[column] = best_answer
        
        return results
    
    def similarity_score(self, token_list_1: List[str], token_list_2: List[str]) -> int:
        """Calculates the numer of matches between two sets of tokens"""

        score = 0

        for token1 in token_list_1:
            for token2 in token_list_2:
                if token1 == token2: score += 1
        
        return score
