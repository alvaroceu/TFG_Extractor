from core.extractor_base import ExtractorBase
from core.preprocessing import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Dict
import numpy as np
import time

class TfidfExtractor(ExtractorBase):

    def extract(self, text: str, questions: str) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Extract relevant information from text for each column."""

        results = {}
        times = {}

        preprocessed_sentences = preprocess(text)
        bags_of_words = preprocess_questions(questions)

        original_sentences = [sentence for sentence, sentence_tokens in preprocessed_sentences]
        sentences_tokens = [" ".join(sentence_tokens) for sentence, sentence_tokens in preprocessed_sentences]
        questions_tokens = [" ".join(question_tokens) for question_tokens in bags_of_words.values()]

        # Build TF-IDF of sentences and questions
        vectorizer = TfidfVectorizer()
        vectorizer.fit(sentences_tokens)

        sentences_tfidf = vectorizer.transform(sentences_tokens)
        questions_tfidf = vectorizer.transform(questions_tokens)

        for index, (column, _) in enumerate(bags_of_words.items()):
            start_time = time.perf_counter()
            best_sentence = self.cosine_similarity_score(questions_tfidf[index], sentences_tfidf,original_sentences)
            times[column] = time.perf_counter() - start_time
            
            results[column] = best_sentence

        return results, times

    def cosine_similarity_score(self, question_tfidf, sentences_tfidf, sentences) -> str:
        """Uses cosine similarity to select the sentence that best answers each question"""

        similarities = cosine_similarity(question_tfidf, sentences_tfidf)
        best_index = np.argmax(similarities)
        
        return sentences[best_index]