import numpy as np

class LogisticRegression:
    """
    Prediction: y_pred = 1 / (1 + e ^(-z))
    Loss Function: Cross Entropy ( + L2 Regularization)
    Gradient: (y_pred - y) * x ( + lambda / n * W
    """
    def __init__(self, lr=0.01, n_iter=1000, reg_lambda=0.0, tol=1e-6):
        self.lr = lr
        self.n_iter = n_iter
        self.reg_lambda = reg_lambda
        self.tol = tol # for early stop
        self.weights = None
        self.b = 0


    def __init_weights(self, n_features):
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=n_features)

    def __sigmoid(self, z):
        # prevent overflow
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def predict_prob(self, X):
        z = X @ self.weights + self.b
        return self.__sigmoid(z)

    def predict(self, X, threshold=0.5):
        prob = self.predict_prob(X)
        return (prob > threshold).astype(int)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.__init_weights(n_features)

        for _ in range(self.n_iter):
            # predict
            y_pred = self.predict_prob(X)

            # grad
            dy = (y_pred - y)
            dw = (1 / n_samples) * dy @ X + (self.reg_lambda / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update
            prev_weights = self.weights.copy()
            self.weights -= self.lr * dw
            self.b -= self.lr * db

            # Early stopping if convergence
            # L2 norm < self.tol
            if np.linalg.norm(self.weights - prev_weights) < self.tol:
                break