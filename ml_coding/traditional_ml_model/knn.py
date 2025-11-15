import numpy as np
from collections import Counter
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X = None  # (n, d)
        self.y = None  # (n, 1)

    def fit(self, X, y):
        self.X = X
        self.y = y

    def _compute_distances(self, X):
        """

        :param X: (m, d), self.X: (n, d)
        :return: distance matrix(m, n)
        """
        return np.linalg.norm(X[:, np.newaxis] - self.X, axis=2)

    def predict(self, X):
        dists = self._compute_distances(X) # (m, n)
        k_idx = np.argsort(dists, axis=1)[:, :self.k]  # (m, k)

        # gather labels
        k_labels = self.y[k_idx]

        # majority vote
        # TODO: check the underlying logic later
        unique_labels = np.unique(self.y)
        counts = (k_labels[..., None] == unique_labels).sum(axis=1)
        preds = unique_labels[np.argmax(counts, axis=1)]
        return preds

# TODO: add ANN implementation