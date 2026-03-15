import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from core.model_evaluator import ModelEvaluator
from core.file_utils import read_databases_json
from traditional_bow.bow_extractor import BoWExtractor
from traditional_bow.tfidf_extractor import TfidfExtractor
from traditional_embeddings.embed_extractor import EmbedExtractorGloVe
from LSTM_methods.use_dan_extractor import USEDANExtractor

def calibrate_thresholds(models_dict, dataset_path="data/squad/parsed_calibration_squad.json"):
    dataset = read_databases_json(dataset_path)
    evaluator = ModelEvaluator()
    best_thresholds = {}
    best_scores = {}

    thresholds_to_test = np.arange(0.0, 1.0, 0.05) # From 0.0 a 0.95 by 0.05 steps

    for model_name, model in models_dict.items():
        print(f"Calibrating {model_name}...")
        best_score = -1
        best_thr = 0.0

        for t in thresholds_to_test:
            model.threshold = t
            all_preds, all_refs = [], []

            for item in dataset:
                text, questions, ground_truths = item["text"], item["questions"], item["ground_truths"]
                results_dict = model.extract(text, questions)
                
                for q_id, ref in ground_truths.items():
                    all_refs.append(ref)
                    all_preds.append(results_dict.get(q_id))

            metrics = evaluator.evaluate_model(all_preds, all_refs)
            
            # Look for the best accuracy proportion
            hasans_acc = metrics.get("HasAns: Accuracy", 0.0)
            noans_acc = metrics.get("NoAns: Accuracy", 0.0)
            score = (hasans_acc + noans_acc) / 2

            if score > best_score:
                best_score = score
                best_thr = t
        
        best_thresholds[model_name] = best_thr
        best_scores[model_name] = best_score

    for model_name in best_thresholds:
        print(f"Best threshold for {model_name}: {best_thresholds[model_name]:.2f} (Score: {best_scores[model_name]:.3f})")
    return best_thresholds

if __name__ == "__main__":

    models = {
        'BoW': BoWExtractor(),
        'tf-idf': TfidfExtractor(),
        'Embeddings gloVe': EmbedExtractorGloVe(),
        'UseDanLSTM': USEDANExtractor(),
    }

    calibrate_thresholds(models)