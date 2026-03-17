from traditional_bow.bow_extractor import BoWExtractor
from traditional_bow.tfidf_extractor import TfidfExtractor
from traditional_embeddings.embed_extractor import EmbedExtractorGloVe
from transformer_method.transformer_bert import TransformerBertExtractor
from transformer_method.transformer_distilbert import TransformerDistilBertExtractor
from LSTM_methods.use_dan_extractor import USEDANExtractor
from core.model_evaluator import *
from core.file_utils import *
from core.export_utils import *
from core.cache_data import warmup_preprocessing_cache
import numpy as np

def main():
    dataset_squad = read_databases_json("data/squad/parsed_squad.json")
    dataset_newsqa = read_databases_json("data/newsqa/parsed_newsqa.json")
    dataset_triviaqa = read_databases_json("data/triviaqa/parsed_triviaqa.json")
    dataset_naturalquestions = read_databases_json("data/natural_questions/parsed_naturalquestions.json")
    
    models = {
        'BoW': BoWExtractor(),
        'tf-idf': TfidfExtractor(),
        'Embeddings gloVe': EmbedExtractorGloVe(),
        'UseDanLSTM': USEDANExtractor(),
        'Transformer DistilBERT': TransformerDistilBertExtractor(),
        'Transformer BERT': TransformerBertExtractor()
    }
    
    evaluator = ModelEvaluator()

    all_predictions = {name: [] for name in models.keys()}
    all_references = []
    all_metrics = {}
    all_times = {name: [] for name in models.keys()}

    # Compute results
    print("Computing results...")
    for item in dataset_squad[:20]:
        text = item["text"]
        questions = item["questions"]
        ground_truths = item["ground_truths"]

        # Compute preprocessing to store data in cache
        warmup_preprocessing_cache(text, questions)

        for q_id, ref in ground_truths.items():
            all_references.append(ref)

        for name, model in models.items():
            results_dict, exec_time = model.timed_extract(text, questions)
            all_times[name].append(exec_time)

            for q_id in ground_truths.keys():
                pred = results_dict.get(q_id, "")
                all_predictions[name].append(pred)

    # Evaluate results
    print("Evaluating results...")
    for name in models.keys():
        metrics = evaluator.evaluate_model(all_predictions[name], all_references)
        
        model_times = all_times[name]
        metrics["Average Time (s) per context"] = np.mean(model_times)
        metrics["Max Time (s)"] = np.max(model_times)
        metrics["Total Time (s)"] = np.sum(model_times)
        
        all_metrics[name] = metrics
    
    # Export results and metrics
    print("Exporting to Excel...")
    export_predictions_to_excel(all_predictions, all_references, "resultado.xlsx")
    export_metrics_to_excel(all_metrics, "metricas.xlsx")
    print("Export Complete")

if __name__ == "__main__":
    main()
