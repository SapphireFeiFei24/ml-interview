import math
import numpy as np
"""
Calculate Rank Related Metrics in numpy
Input: Predictions, Labels
Output: score
"""
class RankMetricsNumpy:
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

    def auc(self, y_true, y_pred):
        # Sort by predicted score descending
        order = np.argsort(-y_pred)
        labels = np.array(y_true)[order]
        height = 0  # num of pos seen so far
        area = 0  # area under curve
        width = 0  # num of neg seen so far
        for l in labels:
            if l == 1:
                height += 1
            else:
                width += 1
                area += height
        return area / (height * width)

    def auc_with_ties(self, predictions, labels):
        """
        Ties: when a positive and a negative have equal prediction values, they're not clearly "ranked"
        AUC = P(scores_pos > score_neg) + 0.5 x P(score_pos == score_neg)
        :param predictions:
        :param labels:
        :return:
        """
        # Sort by predictions descending
        order = np.argsort(-predictions)
        preds_sorted = np.array(predictions)[order]
        labels_sorted = np.array(labels)[order]

        n_pos = np.sum(labels_sorted)
        n_neg = len(labels_sorted) - n_pos

        # Handle edge cases
        if n_pos == 0 or n_neg == 0:
            return 0.0

        area = 0
        height = 0  # positives seen so far
        i = 0
        n = len(preds_sorted)

        # Traverse sorted predictions
        while i < n:
            # group equal scores (ties)
            j = i
            while j < n and preds_sorted[j] == preds_sorted[i]:
                j += 1

            # count positives and negatives in this tie group
            pos_in_group = np.sum(labels_sorted[i:j])
            neg_in_group = (j - i) - pos_in_group

            # Each negative seen so far (including this tie) adds to area
            # Positives before this group count fully,
            # Positives within the tie group count as 0.5
            area += height * neg_in_group + 0.5 * pos_in_group * neg_in_group

            # update total positives seen
            height += pos_in_group
            i = j

        auc_value = area / (n_pos * n_neg)
        return auc_value
