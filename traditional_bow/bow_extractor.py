from core.extractor_base import ExtractorBase
from core.preprocessing import preprocess
from core.preprocessing import preprocess_questions
from typing import List

class BoWExtractor(ExtractorBase):

    def __init__(self, threshold: float = 0.35):

        self.threshold = threshold

    def extract(self, text: str, questions: str):
        """Extract relevant information from text for each column."""
        
        results = {}

        preprocessed_sentences = preprocess(text)
        bags_of_words = preprocess_questions(questions)

        for column, bag_of_words in bags_of_words.items():
            best_score = 0

            for sentence, sentence_tokens in preprocessed_sentences:

                score = self.similarity_score(sentence_tokens, bag_of_words)
                if score > best_score:
                    best_score = score
                    best_answer = sentence

            if best_score < self.threshold:

                best_answer = "A possible valid answer wasn't found"

            results[column] = best_answer
        
        return results
    
    def similarity_score(self, token_list_1: List[str], token_list_2: List[str]) -> int:
        """Compute F1-score based similarity between two sets of tokens."""

        set1 = set(token_list_1)
        set2 = set(token_list_2)
        intersection = set1 & set2
        precision = len(intersection) / len(set2) if set2 else 0
        recall = len(intersection) / len(set1) if set1 else 0

        if precision + recall == 0:
            return 0
        
        score = 2 * (precision * recall) / (precision + recall)

        return score