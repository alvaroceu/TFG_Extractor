import spacy
from core.extractor_base import ExtractorBase
from core.preprocessing import preprocess, parse_questions_embeddings

model = spacy.load("en_core_web_lg")

class EmbedExtractorGloVe(ExtractorBase):

    def extract(self, text: str, questions: str):
        """Extract relevant information from text for each column using static embeddings."""

        results = {}

        preprocessed_sentences = preprocess(text)
        parsed_questions = parse_questions_embeddings(questions)

        sentence_gloVe_vectors = []
        for original_sentence, sentence_tokens in preprocessed_sentences:

            join_tokens = " ".join(sentence_tokens)
            vector = model(join_tokens)
            sentence_gloVe_vectors.append((original_sentence, vector))

        for key, question in parsed_questions.items():

            preprocessed_question = preprocess(question)
            join_question_tokens = " ".join(preprocessed_question[0][1])
            question_gloVe_vector = model(join_question_tokens)

            best_sentence = self.cosine_similarity_score(question_gloVe_vector, sentence_gloVe_vectors)
            results[key] = best_sentence
        
        return results

    def cosine_similarity_score(self, question_gloVe_vector, sentence_gloVe_vectors) -> str:
        """Uses SpaCy's built-in cosine similarity to select the sentence that best answers each question."""

        best_score = 0.0
        best_sentence = "A possible valid answer wasn't found"

        for original_sentence, sentence_doc in sentence_gloVe_vectors:
            
            score = question_gloVe_vector.similarity(sentence_doc)
            
            if score > best_score:
                best_score = score
                best_sentence = original_sentence

        if best_score < 0.4:
            return "A possible valid answer wasn't found"
        
        return best_sentence