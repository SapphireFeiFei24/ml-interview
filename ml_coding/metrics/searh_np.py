import math
import numpy as np
"""
Calculate Search Related Metrics in numpy
Input: Predictions, Labels
Output: score
"""
class SearchMetricsNumpy:
    def __init__(self):
        pass
    def precision_at_k(self, y_true, y_pred, k):
        y_pred_k = y_pred[:k]
        rel = sum(1 for doc in y_pred_k if doc in y_true)
        return rel / k

    def recall_at_k(self, y_true, y_pred, k):
        y_pred_k = y_pred[:k]
        rel = sum(1 for doc in y_pred_k if doc in y_true)
        return rel / len(y_true)

    def average_precision(self, y_true, y_pred):
        hits, precisions = 0, []
        for i, doc in enumerate(y_pred):
            if doc in y_true:
                hits += 1
                precisions.append(hits / (i + 1))
        return np.mean(precisions) if precisions else 0.0

    def mean_reciprocal_rank(self, list_of_queries):
        rr = []
        for y_true, y_pred in list_of_queries:
            for i, doc in enumerate(y_pred, start=1):
                if doc in y_true:
                    rr.append(1/i)
        return np.mean(rr) if rr else 0.0