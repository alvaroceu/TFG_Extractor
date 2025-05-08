from abc import ABC, abstractmethod
from typing import List, Dict

class ExtractorBase(ABC):

    @abstractmethod
    def extract(self, text: str, bags_of_words: Dict[str, List[str]]):
        """Extract relevant information from text for each column."""
        pass
