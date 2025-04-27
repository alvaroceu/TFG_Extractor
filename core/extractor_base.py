from abc import ABC, abstractmethod
from typing import List

class ExtractorBase(ABC):

    @abstractmethod
    def extract(self, text: str, columns: List[str]):
        """Extract relevant information from text for each column."""
        pass
