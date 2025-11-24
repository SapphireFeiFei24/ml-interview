import numpy as np
"""
Assumption:
    Every feature is conditionally independent from each other
    
Mathematical Formular
    argmaxP(c|x1, x2, ..., xd)
    P(c|x) = P(c) * prod_over_x(P(x|c)) / P(x)
    P(x) is consistent across classes:
        only need to find argmax P(c) * prod_over_x(P(x|c))
    To avoid underflow from products:
        find argmax logP(c) + sum_over_x(log(P(x|c)))
"""

class MultinomialNaiveBayes:
    """
    Usage: text classification
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # for laplace smoothing
        self.class_log_prior = None
        self.feature_log_prob = None
        self.classes = None

    def fit(self, X, y):
        """

        :param X: (n_samples, n_features)
        :param y: (n_samples, )
        :return:
        """
        self.classes, class_counts = np.unique(y, return_counts=True)
        n_samples, n_features = X.shape

        # Prior: log P(c)
        self.class_log_prior = np.log(class_counts / n_samples)

        # Likelihood: log(P(x|c))
        feature_count = np.zeros((len(self.classes), n_features))

        for idx, cls in enumerate(self.classes):
            feature_count[idx] = np.sum(X[y == cls]).sum(axis=0)

        # Laplace smoothing
        smoothed_fc = feature_count + self.alpha
        smoothed_total = smoothed_fc.sum(axis=1).reshape(-1, 1)

        # Log prob of features
        self.feature_log_prob = np.log(smoothed_fc / smoothed_total)

    def predict(self, X):
        """

        :param X: (n_samples, n_features)
        :return:
        """
        log_probs = self.class_log_prior + X @ self.feature_log_prob.T

        class_indices = np.argmax(log_probs)
        return self.classes[class_indices]

    def predict_prob(self, X):
        """
        Return the probability of each class
        :param X:
        :return:
        """
        log_probs = self.class_log_prior + X @ self.feature_log_prob.T
        probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        return probs / probs.sum(axis=1, keepdims=True)