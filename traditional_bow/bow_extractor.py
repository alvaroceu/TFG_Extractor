from core.extractor_base import ExtractorBase
from traditional_bow.preprocessing import preprocess
from typing import List, Dict

class BoWExtractor(ExtractorBase):

    def extract(self, text: str, bags_of_words: Dict[str, List[str]]):
        """Extract relevant information from text for each column."""
        
        results = {}

        paragraphs = self.split_text_into_paragraphs(text) 
        preprocessed_paragraphs = [(paragraph, preprocess(paragraph)) for paragraph in paragraphs]

        for column, bag_of_words in bags_of_words.items():
            best_score = 0
            best_answer = "A possible valid answer wasn't found"

            for paragraph, paragraph_tokens in preprocessed_paragraphs:

                score = self.similarity_score(paragraph_tokens, bag_of_words)
                if score > best_score:
                    best_score = score
                    best_answer = paragraph

            results[column] = best_answer
        
        return results

    
    def split_text_into_paragraphs(self, text: str) -> List[str]:
        """Divides the content of the input document in the original paragraphs"""

        paragraphs = [paragraph.strip() for paragraph in text.split('\n') if paragraph.strip()]
        return paragraphs
    
    def similarity_score(self, token_list_1: List[str], token_list_2: List[str]) -> int:
        """Calculates the numer of matches between two sets of tokens"""

        score = 0

        for token1 in token_list_1:
            for token2 in token_list_2:
                if token1 == token2: score += 1
        
        return score
