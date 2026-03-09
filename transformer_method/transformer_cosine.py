from sentence_transformers import SentenceTransformer
from core.extractor_base import ExtractorBase
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from core.preprocessing import *

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

class TransformerCosineExtractor(ExtractorBase):

    def extract(self, text: str, questions: str):
        """Extract relevant information from text for each column."""

        results = {}

        sentences = split_text_into_sentences(text)
        sentence_embeddings = model.encode(sentences)
        parsed_questions = parse_questions_embeddings(questions)

        for key, question in parsed_questions.items():
            question_embedding = model.encode([question])[0]
            best_sentence = self.cosine_similarity_score(question_embedding, sentence_embeddings, sentences)
            results[key] = best_sentence
        
        return results

    def cosine_similarity_score(self, question_embedding, sentence_embeddings, sentences) -> str:
        """Uses cosine similarity to select the sentence that best answers each question"""

        similarities = cosine_similarity([question_embedding], sentence_embeddings)[0]
        best_index = np.argmax(similarities)
        best_score = similarities[best_index]
        print(best_score)

        if best_score < 0.4:
            return "A possible valid answer wasn't found"
        return sentences[best_index]