import numpy as np

class LinearRegression:
    """
    Y = X*W + B
    Loss = 1/(2n) * sum((y-y_pred)**2) + lambda/(2n)*W^2
    Gradient: (y_pred - y) * x
    """
    def __init__(self, lr=0.01, n_iters=1000, reg_lambda=0.0):
        self.weights = None
        self.bias = 0
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.n_iters = n_iters

    def __init_weights(self, n_features):
        """
        Random init weights based on feature dim.
        Avoid init all zeros. That'll make the update of all weights to the same direction
        :param n_features:
        :return:
        """
        self.weights = np.random.normal(loc=0.0, scale=1.0, size=n_features)

    def predict(self, X):
        return X @ self.weights + self.bias

    def fit(self, X, y):
        """
        :param X: (n_samples, n_features)
        :param y: (n_samples)
        :return:
        """
        n_samples, n_features = X.shape
        self.__init_weights(n_features)

        for _ in range(self.n_iters):
            # Prediction
            y_pred = self.predict(X)

            # Calc Gradients
            dy = (y_pred - y)
            dw = (1/n_samples) * (X.T @ dy) + (self.reg_lambda/n_samples) * self.weights
            db = (1/n_samples) * np.sum(dy)

            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
