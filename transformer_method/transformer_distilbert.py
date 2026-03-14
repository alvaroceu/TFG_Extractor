from transformers import pipeline
from core.extractor_base import ExtractorBase
from core.preprocessing import *

class TransformerDistilBertExtractor(ExtractorBase):

    def __init__(self):

        self.model = pipeline('question-answering', model='twmkn9/distilbert-base-uncased-squad2', tokenizer='twmkn9/distilbert-base-uncased-squad2', device=0)

    def extract(self, text: str, questions: str):
        """Extract relevant information from text for each column."""

        results = {}

        parsed_questions = parse_questions_embeddings(questions)

        for key, question in parsed_questions.items():
            
            result = self.model(question=question,context=text,handle_impossible_answer=True)

            if result['answer'].strip() == "":
                best_answer = "A possible valid answer wasn't found"
            else:
                best_answer = result['answer']

            results[key] = best_answer
            print(f"{key}: score={result['score']:.3f} | answer={best_answer}")

        return results