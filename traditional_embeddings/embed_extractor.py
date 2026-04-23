import time
import spacy
from core.extractor_base import ExtractorBase
from core.preprocessing import preprocess, parse_questions_embeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict

class EmbedExtractorGloVe(ExtractorBase):

    def __init__(self):
        # We load the model, but we disable all preprocessing components since we will handle that ourselves
        self.model = spacy.load("en_core_web_lg", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])

    def extract(self, text: str, questions: str) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Extract relevant information from text using static embeddings and sklearn."""

        results = {}
        times = {}

        preprocessed_sentences = preprocess(text)
        parsed_questions = parse_questions_embeddings(questions)

        original_sentences = [sentence for sentence, _ in preprocessed_sentences]

        sentence_vectors = []
        for _, sentence_tokens in preprocessed_sentences:
            join_tokens = " ".join(sentence_tokens)
            vector = self.model(join_tokens).vector
            sentence_vectors.append(vector)
        
        sentence_vectors = np.array(sentence_vectors)

        for key, question in parsed_questions.items():
            preprocessed_question = preprocess(question)
            join_question_tokens = " ".join(preprocessed_question[0][1])
            
            start_time = time.perf_counter()
            question_vector = self.model(join_question_tokens).vector
            question_vector_2d = question_vector.reshape(1, -1)
            best_sentence = self.cosine_similarity_score(question_vector_2d, sentence_vectors, original_sentences)
            times[key] = time.perf_counter() - start_time
            
            results[key] = best_sentence
        
        return results, times

    def cosine_similarity_score(self, question_vector, sentence_vectors, sentences) -> str:
        """Uses sklearn cosine similarity to select the sentence that best answers each question."""

        similarities = cosine_similarity(question_vector, sentence_vectors)
        best_index = np.argmax(similarities)
        
        return sentences[best_index]