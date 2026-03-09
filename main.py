from traditional_bow.bow_extractor import BoWExtractor
from traditional_bow.tfidf_extractor import TfidfExtractor
from traditional_embeddings.embed_extractor import EmbedExtractorGloVe
from transformer_method.transformer_bert import TransformerBertExtractor
from transformer_method.transformer_cosine import TransformerCosineExtractor
from LSTM_methods.flair_LSTM_extractor import FlairLSTMExtractor
from core.model_evaluator import *
from core.file_utils import *
from pprint import pprint
from core.export_utils import *
import os

def main():
    dataset_squad = read_databases_json("data/squad/parsed_squad.json")
    dataset_newsqa = read_databases_json("data/newsqa/parsed_newsqa.json")
    dataset_triviaqa = read_databases_json("data/triviaqa/parsed_triviaqa.json")
    dataset_naturalquestions = read_databases_json("data/natural_questions/parsed_naturalquestions.json")
    
    models = {
        'BoW': BoWExtractor(),
        'tf-idf': TfidfExtractor(),
        'Embeddings gloVe': EmbedExtractorGloVe(),
        'BiLSTM': FlairLSTMExtractor(),
        'Transformer Cosine': TransformerCosineExtractor(),
        'Transformer BERT': TransformerBertExtractor()
    }
    
    evaluator = ModelEvaluator()

    all_predictions = {name: [] for name in models.keys()}
    all_references = []
    all_metrics = {}

    # Compute results
    print("Computing results...")
    for item in dataset_naturalquestions[:30]:
        text = item["text"]
        questions = item["questions"]
        ground_truths = item["ground_truths"]

        for q_id, ref in ground_truths.items():
            all_references.append(ref)

        for name, model in models.items():
            results_dict = model.extract(text, questions)
            
            for q_id in ground_truths.keys():
                pred = results_dict.get(q_id, "")
                all_predictions[name].append(pred)

    # Evaluate results
    print("Evaluating results...")
    for name in models.keys():
        metrics = evaluator.evaluate_model(all_predictions[name], all_references)
        all_metrics[name] = metrics
    
    # Export results and metrics
    print("Exporting to Excel...")
    export_predictions_to_excel(all_predictions, all_references, "resultado.xlsx")
    export_metrics_to_excel(all_metrics, "metricas.xlsx")
    print("Export Complete")

if __name__ == "__main__":
    main()
