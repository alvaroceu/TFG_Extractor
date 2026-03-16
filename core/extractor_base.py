from abc import ABC, abstractmethod
from typing import List, Dict
import time
import torch

class ExtractorBase(ABC):

    @abstractmethod
    def extract(self, text: str, questions: str):
        """Extract relevant information from text for each column/question."""
        pass
    
    def timed_extract(self, text: str, questions: str):
        """Extract relevant information from text for each column/question. Includes computation time"""
        
        # Sync if GPU is being used
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()

        results = self.extract(text, questions)

        # Sync if GPU is being used
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        
        return results, execution_time