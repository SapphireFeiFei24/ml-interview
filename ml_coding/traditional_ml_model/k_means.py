import numpy as np

class KMeans:
    """
    Implementation Details
    * Handle empty clusters (reinitialize randomly)
    * Stop when centroid shift < tolerance
    * use random seed for reproducibility
    """
    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.inertia_ = None # Sum of squared distances to nearest centroid

    def _init_centroids(self, X):
        """
        Randomly choose k samples as initial centroids
        :param X:
        :return:
        """

        rng = np.random.default_rng(self.random_state)
        # generate n_clusters indices based on the x count, no replacement
        indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) == 0:
                # handle empty cluster by reinitializing
                new_centroids[k] = X[np.random.randint(0, X.shape[0])]
            else:
                new_centroids[k] = cluster_points.mean(axis=0)
        return new_centroids

    def fit(self, X):
        self.centroids = self._init_centroids(X)

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)
            shift = np.linalg.norm(self.centroids - new_centroids)
            if shift < self.tol:
                break
            self.centroids = new_centroids

        # Compute final inertia
        distances = np.linalg.norm(X[:, np.newaxis] - self.centoids, axis=2)
        closest = np.min(distances, axis=1)
        self.inertia_ = np.sum(closest ** 2)
        self.labels_ = labels
        return self

    def predict(self, X):
        return self._assign_clusters(X)