from traditional_ml_model.linear_regression import LinearRegression
from traditional_ml_model.logistic_regression import LogisticRegression
from metrics.regression import RegressionMetrics
from metrics.classification import ClassificationMetrics

import numpy as np

def test_linear_regression():
    X_train = np.array([[i] * 4 for i in range(5)])
    W = np.random.rand(4)
    b = 4
    y_train = X_train @ W + b

    model = LinearRegression(lr=0.01, n_iters=1000)
    model.fit(X_train, y_train)

    X_test = np.array([[i] * 4 for i in range(6, 10)])
    y_test = X_test @ W + b
    y_pred = model.predict(X_test)
    metrics = RegressionMetrics()
    mse = metrics.mse(y_true=y_test, y_pred=y_pred)
    print(f"y_test:{y_pred}, y_train:{y_train} mse:{mse}")


def test_logistic_regression():
    # Dummy dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(lr=0.1, n_iter=10000)
    model.fit(X, y)

    metrics = ClassificationMetrics()
    print("Weights:", model.weights)
    print("Bias:", model.b)
    y_pred = model.predict(X)
    print("Predictions:", y_pred)
    print("Recall", metrics.recall(y, y_pred))
if __name__ == "__main__":
    ## Linear Regression
    # test_linear_regression()

    ## Logistic Regression
    test_logistic_regression()