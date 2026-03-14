from transformers import pipeline
from core.extractor_base import ExtractorBase
from core.preprocessing import *

#otros posibles: deepset/roberta-base-squad2
model = pipeline('question-answering', model='deepset/bert-large-uncased-whole-word-masking-squad2', tokenizer='deepset/bert-large-uncased-whole-word-masking-squad2', device=0)

class TransformerBertExtractor(ExtractorBase):

    def extract(self, text: str, questions: str):
        """Extract relevant information from text for each column."""

        results = {}

        parsed_questions = parse_questions_embeddings(questions)

        for key, question in parsed_questions.items():
            
            result = model(question=question,context=text,handle_impossible_answer=True)

            if result['answer'].strip() == "":
                best_answer = "A possible valid answer wasn't found"
            else:
                best_answer = result['answer']

            results[key] = best_answer
            print(f"{key}: score={result['score']:.3f} | answer={best_answer}")

        return results
