from traditional_ml_model.linear_regression import LinearRegression
from metrics.regression import RegressionMetrics


import numpy as np
if __name__ == "__main__":
    # Linear Regression
    X_train = np.array([[i] * 4 for i in range(5)])
    W = np.random.rand(4)
    b = 4
    y_train = X_train @ W + b

    model = LinearRegression(lr=0.01, n_iters=1000)
    model.fit(X_train, y_train)

    X_test = np.array([[i] * 4 for i in range(6, 10)])
    y_test = X_test @ W + b
    y_pred = model.predict(X_test)
    print(y_test)
    print(y_pred)
    metrics = RegressionMetrics()
    mse = metrics.mse(y_true=y_test, y_pred=y_pred)
    print("mse", mse)
