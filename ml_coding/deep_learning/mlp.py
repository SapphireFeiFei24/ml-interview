import numpy as np

from activation import ActivationFuncs
from ..metrics.classification import ClassificationMetrics

class Dense:
    def __init__(self, in_dim, out_dim, activation="relu"):
        # randn generate random numbers following a standard normal distribution with mean=0, var=1
        # general very small number centered around 0
        self.W = np.random.randn(in_dim, out_dim) * 0.01
        self.b = np.zeros((1, out_dim))
        self.activation = activation

        self.activation_funcs = ActivationFuncs()

    def forward(self, X):
        """
        Transform: linear + activate
        :param X: (n_samples, dim_in)
        :return: (n_samples, dim_out)
        """
        self.X = X

        # linear transformation
        # (n_samples, dim_in) @ (dim_in, dim_out) -> (n_samples, dim_out)
        self.Z = self.X @ self.W + self.b

        # activation
        A = self.activation_funcs.activate(self.Z)

        return A

    def backward(self, dA, lr):
        """
        Update the weights and bias
        :param dA: (n_samples, dim_out)
        :param lr:
        :return: None
        """
        # (n_samples, dim_out)
        dZ = dA * self.activation_funcs.grad(self.Z)

        # (n_samples, dim_in).T @ (n_samples, dim_out) -> (dim_in, dim_out)
        n_samples = self.X.shape[0]
        dW = self.X.T @ dZ / n_samples
        db = np.sum(dZ, axis=0, keepdims=True) / n_samples  # average over samples
        dX = dZ @ self.W.T

        # SGD update
        self.W -= lr * dW
        self.b -= lr * db
        return dX


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.loss = ClassificationMetrics()

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_pred, y, lr):
        dA = self.loss.grad_cross_entropy(y_pred=y_pred, y=y)
        for layer in reversed(self.layers):
            dA = layer.backward(dA, lr)

    def fit(self, X, y, lr=0.01, epochs=1000):
        for i in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss.cross_entropy(y_pred=y_pred, y=y)
            self.backward(y_pred, y, lr)
            if i % 200 == 0:
                print(f"Epoch {i} Loss={loss:.4f}")

    def predict(self, X):
        y = self.forward(X)
        return (y > 0.5).astype(int)


"""
Use Example
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

nn = NeuralNetwork([
    Dense(2, 4, activation="relu"),
    Dense(4, 1, activation="sigmoid")
])

nn.fit(X, y, lr=0.1, epochs=2000)
print(nn.predict(X))
"""