import numpy
"""
Calculate Classification Related Metrics
"""

class ClassificationMetrics:
    def  __init__(self):
        pass

    """
    Reusable Module
    """
    def confusion_matrics(self, y_true, y_pred):
        tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
        tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
        return tp, tn, fp, fn

    """
    Of all the items that are predicted True, how much percent are actually true
    """
    def precision(self, y_true, y_pred):
        tp, tn, fp, fn = self.confusion_matrics(self, y_true, y_pred)
        return tp / (tp + fp + 1e-10)  # small fraction for no deviding zero

    """
    Of all the real positive items, how much percent are "recalled"
    """
    def recall(self, y_true, y_pred):
        tp, tn, fp, fn = self.confusion_matrics(y_true, y_pred)
        return tp / (tp + fn + 1e-10)

    """
    Definition
    Harmonic mean of precision and recall 
    """
    def f1_score(self, y_true, y_pred):
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        return 2 * precision * recall / (precision + recall + 1e-10)

    def accuracy(self, y_true, y_pred):
        tp, tn, fp, fn = self.confusion_matrics(self, y_true, y_pred)
        return (tp + tn) / (tp + tn + fp + fn)