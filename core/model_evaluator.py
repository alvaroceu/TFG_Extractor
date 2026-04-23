import evaluate
from typing import List, Dict, Any

class ModelEvaluator:
    """Evaluates NLP Question Answering models in BATCH mode for high performance"""

    def __init__(self, dataset_empty_flag: str = "No answer found", model_empty_flag: str = "A possible valid answer wasn't found"):
        self.rouge_metric = evaluate.load("rouge")
        self.bertscore_metric = evaluate.load("bertscore")
        
        self.dataset_empty_flag = dataset_empty_flag
        self.model_empty_flag = model_empty_flag

    def _normalize_text(self, text: str) -> str:
        """Lowercases and strips whitespace for fair lexical comparison."""
        return text.strip().lower()

    def evaluate_batch(self, predictions: List[str], references: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluates a massive list of predictions and references all at once.
        Returns a list of dictionaries with the metrics, keeping the exact same order.
        """
        results = []
        
        # These lists will store only the cases where BERTScore needs to be calculated (True Positives)
        tp_indices = []
        tp_preds = []
        tp_refs = []

        print("  -> Calculating base metrics (Exact Match, Status)...")
        # STEP 1: Calculate basic metrics (Takes < 1 second for 1 million rows)
        for i, (p, r) in enumerate(zip(predictions, references)):
            pred = "" if p == self.model_empty_flag else self._normalize_text(p)
            ref = "" if r == self.dataset_empty_flag else self._normalize_text(r)

            status = ""
            rouge_score = 0.0
            bert_score = 0.0

            if pred == "" and ref == "":
                status = "TN"
            elif pred != "" and ref == "":
                status = "FP"
            elif pred == "" and ref != "":
                status = "FN"
            else:
                status = "TP"
            
            exact_match = 0
            if status == "TP" and pred == ref:
                exact_match = 1

            inclusion_match = 0
            if status == "TP" and ((ref in pred) or (pred in ref)):
                inclusion_match = 1

            if status == "TN":
                inclusion_match = 1
                exact_match = 1
                rouge_score = 1
                bert_score = 1

            # Initialize results with 0.0 for ROUGE and BERTScore
            results.append({
                "Status": status,
                "ExactMatch": exact_match,
                "InclusionMatch": inclusion_match,
                "ROUGE_L": rouge_score,
                "BERTScore": bert_score
            })

            # If it is a True Positive, save it for deep learning evaluation
            if status == "TP":
                tp_indices.append(i)
                tp_preds.append(pred)
                tp_refs.append(ref)

        # STEP 2: Calculate Deep Learning metrics ONLY for True Positives and in BATCH
        if tp_indices:
            print(f"  -> Calculating ROUGE-L for {len(tp_indices)} valid answers...")
            # use_aggregator=False returns individual results, not the mean
            rouge_out = self.rouge_metric.compute(predictions=tp_preds, references=tp_refs, use_aggregator=False)
            rouge_scores = rouge_out['rougeL']

            print(f"  -> Calculating BERTScore for {len(tp_indices)} valid answers (Using GPU Batching)...")
            # batch_size=64 significantly speeds up the process on RTX 2080 Ti
            bert_out = self.bertscore_metric.compute(predictions=tp_preds, references=tp_refs, lang="en", batch_size=64)
            bert_scores = bert_out['f1']

            # STEP 3: Reassign the scores to their corresponding row using indices
            for idx, r_score, b_score in zip(tp_indices, rouge_scores, bert_scores):
                results[idx]["ROUGE_L"] = r_score
                results[idx]["BERTScore"] = b_score

        return results