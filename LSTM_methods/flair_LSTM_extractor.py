from core.extractor_base import ExtractorBase
from core.preprocessing import split_text_into_sentences, parse_questions_embeddings
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')
document_embeddings = DocumentPoolEmbeddings([flair_embedding_forward,flair_embedding_backward])

class FlairLSTMExtractor(ExtractorBase):

    def get_sentence_vector(self, text: str):
        """Helper to pass text through the pre-trained LSTM and get its vector."""

        # Flair uses its own 'Sentence' object wrapper
        sentence = Sentence(text)
        self.document_embeddings.embed(sentence)
        return sentence.embedding.cpu().numpy()

    def extract(self, text: str, questions: str):
        """Extract relevant information from text using Pre-trained LSTM embeddings."""
        
        results = {}
        sentences = split_text_into_sentences(text)
        parsed_questions = parse_questions_embeddings(questions)

        sentence_vectors = [self.get_sentence_vector(s) for s in sentences]
        sentence_embeddings = np.array(sentence_vectors)

        for key, question in parsed_questions.items():
            question_embedding = self.get_sentence_vector(question)
            
            similarities = cosine_similarity([question_embedding], sentence_embeddings)[0]
            best_index = np.argmax(similarities)
            best_score = similarities[best_index]
            
            if best_score < 0.5:
                results[key] = "A possible valid answer wasn't found"
            else:
                results[key] = sentences[best_index]

        return results