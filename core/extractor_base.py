from abc import ABC, abstractmethod
from typing import List, Dict

class ExtractorBase(ABC):

    @abstractmethod
    def extract(self, text: str, questions: str):
        """Extract relevant information from text for each column/question."""
        pass
