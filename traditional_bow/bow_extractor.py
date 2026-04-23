from core.extractor_base import ExtractorBase
from core.preprocessing import preprocess, preprocess_questions
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from typing import Tuple, Dict
import time

class BoWExtractor(ExtractorBase):

    def extract(self, text: str, questions: str) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Extract relevant information from text for each column."""
        
        results = {}
        times = {}

        preprocessed_sentences = preprocess(text)
        bags_of_words = preprocess_questions(questions)

        original_sentences = [sentence for sentence, sentence_tokens in preprocessed_sentences]
        sentences_tokens = [" ".join(sentence_tokens) for sentence, sentence_tokens in preprocessed_sentences]
        questions_tokens = [" ".join(question_tokens) for question_tokens in bags_of_words.values()]
        
        # Build BoW vectors of sentences and questions
        vectorizer = CountVectorizer()
        vectorizer.fit(sentences_tokens)

        sentences_bow = vectorizer.transform(sentences_tokens)
        questions_bow = vectorizer.transform(questions_tokens)

        for index, (column, _) in enumerate(bags_of_words.items()):
            start_time = time.perf_counter()
            best_sentence = self.cosine_similarity_score(questions_bow[index], sentences_bow,original_sentences)
            times[column] = time.perf_counter() - start_time
            
            results[column] = best_sentence

        return results, times

    def cosine_similarity_score(self, question_bow, sentences_bow, sentences) -> str:
        """Uses cosine similarity to select the sentence that best answers each question"""

        similarities = cosine_similarity(question_bow, sentences_bow)
        best_index = np.argmax(similarities)
        
        return sentences[best_index]