import math
"""
Calculate Rank Related Metrics
Input: Predictions, Labels
Output: score
"""
class RankMetrics:
    def __init__(self):
        pass

    """
    Definition:
        Out of the top K returned result, how many are relevant
    Input:
        Predictions: Ranked results
        Labels: label i is the GT for the prediction at i, 1 as positive
    """
    def precision_at_k(self, predictions, labels, k):
        rel = 0
        for i in range(k):
            if labels[i] == 1:
                rel += 1
        return rel / k

    """
    Definition:
        Out of the all relavant items, how many are at top K
    Input:
        Predictions: Ranked results
        Labels: label i is the GT for the prediction at i, 1 as positive
    """
    def recall_at_k(self, predictions, labels, k):
        total = sum(labels)
        rel = 0
        for i in range(k):
            if labels[i] == 1:
                rel += 1
        return rel / total

    """
    Definition:
        Average of Precision@K at every position where a rel item appears
    """
    def average_precision(self, predictions, labels):
        sum_pk = 0
        rel = 0
        hits = 0
        for k in range(len(labels)):
            if labels[k] == 1:
                hits += 1
                pk = hits / (k + 1)
                sum_pk += pk
                rel += 1
        return sum_pk / rel

    """
    Definition:
        Mean Averaged Precision
        For each relevant position k, cal P@K
        avg over all position, over all data
    Input:
        List of queries
    """
    def mean_average_precision(self, prediction_list, label_list):
        sum_ap = 0
        for q in range(len(prediction_list)):
            ap = self.average_precision(prediction_list[q], label_list[q])
            sum_ap += ap
        return sum_ap / len(prediction_list)

    """
    Definition
    Mean Reciprocal Rank
    Focuses on how early the "first" rel item appears
    Averaged over all queries
    """
    def mean_reciprocal_rank(self, prediction_list, label_list):
        sum_rank = 0
        for q in range(len(prediction_list)):
            prediction, label = prediction_list[q], label_list[q]
            r = 0
            for i in range(len(label)):
                if label[i] == 1:
                    r = 1 / (i+1)
                    break
            sum_rank += r
        return sum_rank / len(prediction_list)

    """
    Definition
    Normalized Discounter Cumulative Gain
    DCG = sum((2^rel-1) / log2(i + 1))
    NDCG = DCG / maxDCG
    
    Input:
    Labels: more fine-grained, not just binary
    """
    def NDCG(self, predictions, labels):
        def dcg(labels):
            total = 0
            for i in range(len(labels)):
                total += (2 ** labels[i] - 1) / math.log2(i + 2)  # to avoid dividing by zero
            return total
        pred_dcg = dcg(labels)
        optimal_labels = sorted(labels, reverse=True)
        max_dcg = dcg(optimal_labels)
        return pred_dcg / max_dcg

    """
    Area Under the ROC Curve
    TPR / FPR
    """
    def auc(self, predictions, labels):
        height = 0
        width = 0
        area = 0
        for l in labels:
            if l == 1:
                height += 1
            else:
                width += 1
                area += height
        return area / (width * height)