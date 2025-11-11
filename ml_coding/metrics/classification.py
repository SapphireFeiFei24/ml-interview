import numpy as np

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

    def cross_entropy(self, y_true, y_pred, eps=1e-15):
        """
        :param y_true: array, 0 or 1
        :param y_pred: array, probability of it to be label 1
        :param eps: small val to avoid log(0)
        :return: mean binary cross-entropy loss
        """
        y_true = np.array(y_true)
        y_pred = np.clip(y_pred, eps, 1-eps) # prevent log(0)
        loss = -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
        return loss

    def softmax(self, logits):
        """
        :param logits: raw model outputs of shape (N, C)
        :return: softmax scores of shape (N, 1) - a probability
        """
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def multiclass_cross_entropy(self, y_true, logits, eps=1e-15):
        probs = self.softmax(logits)
        probs = np.clip(probs, eps, 1-eps)
        loss = -np.mean(np.sum(y_true * np.log(probs), axis=1))
        return loss
