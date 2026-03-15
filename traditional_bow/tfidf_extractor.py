from core.extractor_base import ExtractorBase
from core.preprocessing import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TfidfExtractor(ExtractorBase):

    def __init__(self, threshold: float = 0.35):

        self.threshold = threshold

    def extract(self, text: str, questions: str):
        """Extract relevant information from text for each column."""

        results = {}

        preprocessed_sentences = preprocess(text)
        bags_of_words = preprocess_questions(questions)

        original_sentences = [sentence for sentence, sentence_tokens in preprocessed_sentences]
        sentences_tokens = [" ".join(sentence_tokens) for sentence, sentence_tokens in preprocessed_sentences]
        questions_tokens = [" ".join(question_tokens) for question_tokens in bags_of_words.values()]

        # Build TF-IDF of sentences and questions
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences_tokens + questions_tokens)

        N = len(sentences_tokens)
        #First N rows --> TF-IDF vectors of sentences
        sentences_tfidf = tfidf_matrix[:N]
        #Remaining rows --< TF-IDF vectors of questions
        questions_tfidf = tfidf_matrix[N:]

        for index, (column, _) in enumerate(bags_of_words.items()):
            best_sentence = self.cosine_similarity_score(questions_tfidf[index], sentences_tfidf,original_sentences)
            results[column] = best_sentence

        return results

    def cosine_similarity_score(self, question_tfidf, sentences_tfidf, sentences) -> str:
        """Uses cosine similarity to select the sentence that best answers each question"""

        similarities = cosine_similarity(question_tfidf, sentences_tfidf)

        best_score = np.max(similarities)
        best_index = np.argmax(similarities)
        
        if best_score < self.threshold:
            return "A possible valid answer wasn't found"

        return sentences[best_index]