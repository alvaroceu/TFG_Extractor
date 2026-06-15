from traditional_bow.bow_extractor import BoWExtractor
from traditional_bow.tfidf_extractor import TfidfExtractor
from traditional_embeddings.embed_extractor import EmbedExtractorGloVe
from transformer_method.BERTLarge import TransformerBertExtractor
from transformer_method.DistilBERT import TransformerDistilBertExtractor
from transformer_method.SparseDistilBERT import TransformerSparseDistilBertExtractor
from transformer_method.SparseBERTLarge import TransformerSparseBertLargeExtractor
from LSTM_methods.use_dan_extractor import USEDANExtractor
from transformer_method.BiLBERT.BiLBERT_Distil import TransformerBiLBERTDistilExtractor
from transformer_method.BiLBERT.BiLBERT_Large import TransformerBiLBERTLargeExtractor

from core.model_evaluator import ModelEvaluator
from core.file_utils import read_databases_json
from core.export_utils import export_results_to_excel
from core.cache_data import warmup_preprocessing_cache

def parse_questions_string_to_dict(questions_str: str) -> dict:
    """Utility to map 'Q1' -> 'Q1: Where is...?' for the Excel row."""
    lines = questions_str.strip().split('\n')
    return {line.split(':')[0].strip(): line.strip() for line in lines if ':' in line}

def main():
    datasets = [
        ("SQuAD 2.0", read_databases_json("data/squad/parsed_squad.json")),
        ("NewsQA", read_databases_json("data/newsqa/parsed_newsqa.json")),
        ("Natural Questions", read_databases_json("data/natural_questions/parsed_naturalquestions.json")),
        ("TriviaQA", read_databases_json("data/triviaqa/parsed_triviaqa.json"))
    ]
    
    models = {
        'BoW': BoWExtractor(),
        'tf-idf': TfidfExtractor(),
        'Embeddings gloVe': EmbedExtractorGloVe(),
        'UseDanLSTM': USEDANExtractor(),
        'Transformer DistilBERT': TransformerDistilBertExtractor(),
        'Transformer BERT': TransformerBertExtractor(),
        'Transformer SparseDistilBERT': TransformerSparseDistilBertExtractor(block_size=64),
        'Transformer SparseBERTLarge': TransformerSparseBertLargeExtractor(block_size=64),
        'Transformer BiLBERTDistil': TransformerBiLBERTDistilExtractor(),
        'Transformer BiLBERTLarge': TransformerBiLBERTLargeExtractor(),
    }
    
    evaluator = ModelEvaluator()
    results = []
    global_question_id = 1

    print("Starting the general processing...")
    
    for ds_name, dataset_content in datasets:
        print(f"\n--- Processing Dataset: {ds_name} ---")
        
        for item in dataset_content:
            text = item["text"]
            questions_str = item["questions"]
            ground_truths = item["ground_truths"]

            # Store preprocessing in cache to avoid the first model having to compute it from scratch (unjust comparison)
            warmup_preprocessing_cache(text, questions_str)
            
            # Map Q_id to the question
            question_texts = parse_questions_string_to_dict(questions_str)

            # Extract results and times for ALL models for this context
            context_results = {}
            context_times = {}
            for model_name, model in models.items():
                res_dict, times_dict = model.extract(text, questions_str)
                context_results[model_name] = res_dict
                context_times[model_name] = times_dict

            # Organize results
            for q_id, ref in ground_truths.items():
                current_global_id = f"ID_{global_question_id}"
                q_text = question_texts.get(q_id, f"Missing text for {q_id}")
                
                # Save results of the 6 models for this question
                for model_name in models.keys():
                    ans = context_results[model_name].get(q_id, "A possible valid answer wasn't found")
                    exec_time = context_times[model_name].get(q_id, 0.0)
                    
                    # Create row of results for this question and model
                    row = {
                        "Question ID": current_global_id,
                        "Dataset": ds_name,
                        "Context": text,
                        "Question": q_text,
                        "Ground Truth": ref,
                        "Model": model_name,
                        "Answer": ans,
                        "ExecTime": exec_time,
                    }
                    results.append(row)
                
                # Next question gets a new ID
                global_question_id += 1

    print("\nExtraction completed. Starting Evaluation...")
    
    # Extract all predictions and references into two large lists keeping the order
    all_preds = [row["Answer"] for row in results]
    all_refs = [row["Ground Truth"] for row in results]
    
    # Pass the giant lists to the evaluator to process in batch
    metrics_list = evaluator.evaluate_batch(all_preds, all_refs)
    
    # Reintegrate metrics into our results list
    for i in range(len(results)):
        results[i].update(metrics_list[i])

    print("\nEvaluation completed. Exporting to Excel...")
    export_results_to_excel(results, "tfg_results_mymodels.xlsx")
    print("Export completed successfully.")

if __name__ == "__main__":
    main()
