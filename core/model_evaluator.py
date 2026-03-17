import evaluate
from typing import List, Dict, Any

class ModelEvaluator:
    """Evaluates NLP Question Answering models using different metrics"""

    def __init__(self, dataset_empty_flag: str = "No answer found", model_empty_flag: str = "A possible valid answer wasn't found"):
        
        self.rouge_metric = evaluate.load("rouge")
        self.bertscore_metric = evaluate.load("bertscore")
        
        self.dataset_empty_flag = dataset_empty_flag
        self.model_empty_flag = model_empty_flag

    def _normalize_text(self, text: str) -> str:
        """Lowercases and strips whitespace for fair lexical comparison."""
        return text.strip().lower()

    def evaluate_model(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Computes all evaluation metrics comparing model predictions against ground truth."""
        total_queries = len(references)
        if total_queries == 0:
            return {}

        # Normalize "No answer" flags to standard empty strings ("")
        norm_preds = ["" if p == self.model_empty_flag else p for p in predictions]
        norm_refs  = ["" if r == self.dataset_empty_flag else r for r in references]

        true_positives = 0   # Model answered, answer exists
        true_negatives = 0   # Model abstained, answer didn't exist
        false_positives = 0  # Model answered, answer didn't exist
        false_negatives = 0  # Model abstained, answer exists

        exact_matches = 0
        inclusion_matches = 0
        TP_preds = []
        TP_refs = []

        for p, r in zip(norm_preds, norm_refs):
            preds = self._normalize_text(p)
            refs = self._normalize_text(r)

            # Robustness
            if preds == "" and refs == "":
                true_negatives += 1
            elif preds != "" and refs == "":
                false_positives += 1
            elif preds == "" and refs != "":
                false_negatives += 1
            else:
                true_positives += 1
                
                if preds == refs:
                    exact_matches += 1
                if (refs in preds) or (preds in refs): 
                    inclusion_matches += 1

                # Save TP pairs
                TP_preds.append(preds)
                TP_refs.append(refs)

        # HasAns Metrics
        total_hasans = true_positives + false_negatives
        exact_score = 0.0
        inclusion_score = 0.0
        rougeL_score = 0.0
        bertscore_score = 0.0
        hasans_detection_score = 0.0

        sum_rouge_points = 0.0
        sum_bertscore_points = 0.0

        if total_hasans > 0:
            hasans_detection_score = true_positives / (true_positives + false_negatives)
            exact_score = exact_matches / total_hasans
            inclusion_score = inclusion_matches / total_hasans

            if TP_refs:
                # Compute BERTScore and RougeL in TPs
                rougeL_results = self.rouge_metric.compute(predictions=TP_preds, references=TP_refs)
                bertscore_results = self.bertscore_metric.compute(predictions=TP_preds, references=TP_refs, lang="en")
                
                sum_rouge_points = rougeL_results['rougeL'] * true_positives 
                sum_bertscore_points = sum(bertscore_results['f1'])

            rougeL_score = sum_rouge_points / total_hasans # total_hasans include FN errors
            bertscore_score = sum_bertscore_points / total_hasans

        # NoAns Metrics
        noans_detection_score = 0.0

        if true_negatives + false_positives > 0:
            noans_detection_score = true_negatives / (true_negatives + false_positives)

        # Global/Average Metrics
        avg_exact_match = (exact_matches + true_negatives) / total_queries
        avg_inclusion_match = (inclusion_matches + true_negatives) / total_queries
        avg_rougeL = (sum_rouge_points + true_negatives) / total_queries
        avg_bertscore = (sum_bertscore_points + true_negatives) / total_queries

        # Output
        results = {
            "True Positives": true_positives,
            "True Negatives": true_negatives,
            "False Positives": false_positives,
            "False Negatives": false_negatives,

            "HasAns: ExactMatch": exact_score,
            "HasAns: InclusionMatch": inclusion_score,
            "HasAns: ROUGE_L": rougeL_score,
            "HasAns: BERTScore": bertscore_score,

            "HasAns: Accuracy": hasans_detection_score,
            "NoAns: Accuracy": noans_detection_score,

            "Average Exact Match": avg_exact_match,
            "Average Inclusion Match": avg_inclusion_match,
            "Average ROUGE-L": avg_rougeL,
            "Average BERTScore": avg_bertscore
        }

        return results