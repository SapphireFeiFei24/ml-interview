import numpy as np
class SVM:
    def __init__(self, lr=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weight = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1) # labels are -1, 1
        self.w = np.random.normal(loc=0.0, scale=1.0, size=n_features)
        self.b = 0

        for _ in range(self.n_iters):
            # compute margins
            margins = y_ * (X @ self.w + self.b)

            # hinge loss mask: same as max(0, 1 - margins)
            misclassified = margins < 1

            # gradients: only misclassified will affect gradients
            dw = self.w - self.lambda_param * (y_[misclassified] @ X[misclassified])
            db = -self.lambda_param * self.sum(y_[misclassified])

            # update
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        linear_output = X @ self.w + self.b
        return np.sign(linear_output)


class KernelTransformation:
    def __init__(self, kernel="linear", degree=2, gamma=0.1, coef0=1, n_features=100):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.n_features = n_features
        self.W = None
        self.b = None

    def fit(self, X):
        if self.kernel == "rbf":
            if self.W is not None:
                return  # only init once
            # Random Fourier Features for RBF approximation
            _, n_features = X.shape
            self.W = np.random.normal(scale=np.sqrt(2*self.gamma),
                                      size=(n_features, self.n_features))

            self.b = np.random.uniform(0, 2*np.pi, size=self.n_features)

    def transform(self, X):
        if self.kernel == "linear":
            return X

        if self.kernel == "poly":
            return (X @ X.T + self.coef0) ** self.degree

        if self.kernel == "rbf":
            z = np.dot(X, self.W) + self.b
            return np.sqrt(2 / self.n_features) * np.cos(z)

        raise ValueError("Unknown kernel type")


class KernelSVM:
    def __init__(self, kernel="linear", degree=2, gamma=0.1, coef=1, n_features=100):
        self.transformer = KernelTransformation(kernel, degree, gamma, coef, n_features)

        self.model = SVM()

    def fit(self, X, y):
        self.transformer.fit(X)
        trans_X = self.transformer.transform(X)
        self.model.fit(trans_X, y)

    def predict(self, X):
        trans_X = self.transformer.transform(X)
        return self.model.predict(trans_X)