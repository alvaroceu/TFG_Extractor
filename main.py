from traditional_bow.bow_extractor import BoWExtractor
from traditional_embeddings.embed_extractor import EmbedExtractor
from core.file_utils import read_raw_text
from pprint import pprint
from core.export_utils import *
import os

def main():

    text = read_raw_text("data/example.txt")
    questions = read_raw_text("data/questions.txt")

    extractor = EmbedExtractor()
    results = extractor.extract(text, questions) 

    results_dict = {}
    results_dict[os.path.basename("data/example.txt")] = results
    results_dict["Otra fila"] = results
    
    export_results_to_excel(questions,results_dict,"resultado.xlsx")

if __name__ == "__main__":
    main()
