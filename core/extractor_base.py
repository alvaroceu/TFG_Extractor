from abc import ABC, abstractmethod
from typing import Tuple, Dict

class ExtractorBase(ABC):

    @abstractmethod
    def extract(self, text: str, questions: str) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Extract relevant information from text for each column/question.
        
        Returns:
            results: Dict mapping question keys to answers
            times: Dict mapping question keys to execution times"""
        pass
