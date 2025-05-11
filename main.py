from traditional_bow.bow_extractor import BoWExtractor
from core.file_utils import read_raw_text
from pprint import pprint

def main():

    text = read_raw_text("data/example.txt")
    questions = read_raw_text("data/questions.txt")

    extractor = BoWExtractor()
    results = extractor.extract(text, questions) 
    pprint(results)

if __name__ == "__main__":
    main()
