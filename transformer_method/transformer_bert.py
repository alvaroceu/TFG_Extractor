from transformers import pipeline
from core.extractor_base import ExtractorBase
from core.preprocessing import *
import time
from typing import Tuple, Dict
import torch

class TransformerBertExtractor(ExtractorBase):

    def __init__(self):

        self.model = pipeline('question-answering', model='deepset/bert-large-uncased-whole-word-masking-squad2', tokenizer='deepset/bert-large-uncased-whole-word-masking-squad2', torch_dtype=torch.float16, device=0)

    def extract(self, text: str, questions: str) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Extract relevant information from text for each column."""

        results = {}
        times = {}

        parsed_questions = parse_questions_embeddings(questions)

        for key, question in parsed_questions.items():

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            result = self.model(question=question,context=text,handle_impossible_answer=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times[key] = time.perf_counter() - start_time

            if result['answer'].strip() == "":
                best_answer = "A possible valid answer wasn't found"
            else:
                best_answer = result['answer']

            results[key] = best_answer

        return results, times
