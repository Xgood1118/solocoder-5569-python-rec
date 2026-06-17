import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class Evaluator:
    def __init__(self, top_ks: Optional[List[int]] = None):
        self.top_ks = top_ks or [5, 10, 20]

    def evaluate(self, predictions: Dict[str, List[str]],
                 ground_truth: Dict[str, List[str]],
                 all_items: Optional[List[str]] = None,
                 item_categories: Optional[Dict[str, str]] = None) -> Dict[int, Dict[str, float]]:
        results = {}

        for k in self.top_ks:
            pred_at_k = {u: items[:k] for u, items in predictions.items()}
            metrics = self._compute_metrics(pred_at_k, ground_truth, all_items, item_categories)
            results[k] = metrics

        return results

    def _compute_metrics(self, predictions: Dict[str, List[str]],
                         ground_truth: Dict[str, List[str]],
                         all_items: Optional[List[str]],
                         item_categories: Optional[Dict[str, str]]) -> Dict[str, float]:
        precisions = []
        recalls = []
        f1s = []
        hit_rates = []
        ndcgs = []
        aps = []

        all_recommended = set()
        all_ground_truth_items = set()

        category_diversities = []
        intra_list_similarities = []

        for user_id in predictions:
            if user_id not in ground_truth:
                continue

            pred = predictions[user_id]
            gt = ground_truth[user_id]

            if not gt:
                continue

            tp = len(set(pred) & set(gt))

            precision = tp / len(pred) if pred else 0.0
            recall = tp / len(gt) if gt else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            hit = 1.0 if tp > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            hit_rates.append(hit)

            ndcgs.append(self._ndcg(pred, gt))
            aps.append(self._average_precision(pred, gt))

            all_recommended.update(pred)
            all_ground_truth_items.update(gt)

            if item_categories:
                cats = [item_categories.get(item, 'unknown') for item in pred if item]
                diversity = len(set(cats)) / max(len(cats), 1)
                category_diversities.append(diversity)

        metrics = {
            'precision': np.mean(precisions) if precisions else 0.0,
            'recall': np.mean(recalls) if recalls else 0.0,
            'f1': np.mean(f1s) if f1s else 0.0,
            'hit_rate': np.mean(hit_rates) if hit_rates else 0.0,
            'map': np.mean(aps) if aps else 0.0,
            'ndcg': np.mean(ndcgs) if ndcgs else 0.0,
        }

        if all_items:
            coverage = len(all_recommended) / len(all_items) if all_items else 0.0
            metrics['coverage'] = coverage

        if category_diversities:
            metrics['diversity'] = np.mean(category_diversities)

        metrics['novelty'] = self._compute_novelty(predictions, all_ground_truth_items)

        return metrics

    def _dcg(self, predictions: List[str], ground_truth: List[str]) -> float:
        dcg = 0.0
        for i, item in enumerate(predictions):
            if item in ground_truth:
                dcg += 1.0 / np.log2(i + 2)
        return dcg

    def _ndcg(self, predictions: List[str], ground_truth: List[str]) -> float:
        dcg = self._dcg(predictions, ground_truth)
        ideal = sorted([1.0] * min(len(ground_truth), len(predictions)), reverse=True)
        idcg = sum(v / np.log2(i + 2) for i, v in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    def _average_precision(self, predictions: List[str], ground_truth: List[str]) -> float:
        if not ground_truth:
            return 0.0

        hits = 0
        sum_prec = 0.0

        for i, item in enumerate(predictions):
            if item in ground_truth:
                hits += 1
                sum_prec += hits / (i + 1)

        return sum_prec / len(ground_truth)

    def _compute_novelty(self, predictions: Dict[str, List[str]],
                       all_gt_items: set) -> float:
        novelty_scores = []

        for user_id, recs in predictions.items():
            if not recs:
                continue
            novel_count = sum(1 for item in recs if item not in all_gt_items)
            novelty_scores.append(novel_count / len(recs))

        return np.mean(novelty_scores) if novelty_scores else 0.0

    def compute_auc(self, predictions: Dict[str, List[Tuple[str, float]]],
                    ground_truth: Dict[str, List[str]]) -> float:
        aucs = []

        for user_id in predictions:
            if user_id not in ground_truth:
                continue

            pred_items = [item for item, _ in predictions[user_id]]
            gt_set = set(ground_truth[user_id])

            if len(gt_set) == 0 or len(gt_set) == len(pred_items):
                continue

            positives = [1 if item in gt_set else 0 for item in pred_items]
            n_pos = sum(positives)
            n_neg = len(positives) - n_pos

            if n_pos == 0 or n_neg == 0:
                continue

            rank_sum = sum(i + 1 for i, p in enumerate(positives) if p)
            auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            aucs.append(auc)

        return np.mean(aucs) if aucs else 0.0

    def compute_serendipity(self, predictions: Dict[str, List[str]],
                           ground_truth: Dict[str, List[str]],
                           expected_items: Dict[str, List[str]]) -> float:
        serendipities = []

        for user_id, recs in predictions.items():
            if user_id not in ground_truth:
                continue

            gt_set = set(ground_truth[user_id])
            expected_set = set(expected_items.get(user_id, []))

            surprising = set(recs) & gt_set - expected_set
            serendipity = len(surprising) / max(len(recs), 1)
            serendipities.append(serendipity)

        return np.mean(serendipities) if serendipities else 0.0

    def format_results(self, results: Dict[int, Dict[str, float]]) -> str:
        lines = ["评估指标结果\n" + "=" * 50]

        for k, metrics in sorted(results.items()):
            lines.append(f"\nTop-{k}:")
            for name, value in metrics.items():
                lines.append(f"  {name}: {value:.4f}")

        return "\n".join(lines)
