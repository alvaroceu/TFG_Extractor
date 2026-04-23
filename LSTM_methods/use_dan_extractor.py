import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from core.extractor_base import ExtractorBase
from core.preprocessing import split_text_into_sentences, parse_questions_embeddings
import time
from typing import Tuple, Dict

class USEDANExtractor(ExtractorBase):
    def __init__(self):

        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def extract(self, text: str, questions: str) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Extract relevant information from text using Universal Sentence Encoder."""
        results = {}
        times = {}
        
        sentences = split_text_into_sentences(text)
        parsed_questions = parse_questions_embeddings(questions)

        sentence_embeddings = self.model(sentences).numpy()

        for key, question in parsed_questions.items():

            start_time = time.perf_counter()
            question_embedding = self.model([question]).numpy()
            similarities = cosine_similarity(question_embedding, sentence_embeddings)[0]
            best_index = np.argmax(similarities)
            best_score = similarities[best_index]
            times[key] = time.perf_counter() - start_time

            if best_score < 0.4:
                results[key] = "A possible valid answer wasn't found"
            else:
                results[key] = sentences[best_index]

        return results, times