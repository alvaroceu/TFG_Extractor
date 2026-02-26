from transformers import pipeline
from core.extractor_base import ExtractorBase
from core.preprocessing import *

#otros posibles: deepset/roberta-base-squad2
model = pipeline('question-answering', model='deepset/bert-large-uncased-whole-word-masking-squad2', tokenizer='deepset/bert-large-uncased-whole-word-masking-squad2')

class TransformerBertExtractor(ExtractorBase):

    def extract(self, text: str, questions: str):
        """Extract relevant information from text for each column."""

        results = {}

        sentences = split_text_into_sentences(text)
        parsed_questions = parse_questions_embeddings(questions)

        chunk_size=8
        chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

        for key, question in parsed_questions.items():
            best_answer = "A possible valid answer wasn't found"
            best_score = 0

            for chunk in chunks:
                result = model(question=question,context=chunk)
                if result['score'] > best_score:
                    best_score = result['score']
                    best_answer = result['answer']

            results[key] = best_answer
            print(f"{key}: score={best_score:.3f} | answer={best_answer}")

        return results
