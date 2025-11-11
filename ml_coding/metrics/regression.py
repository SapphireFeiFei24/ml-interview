import numpy as np

"""
Calculate Regression Related Metrics
"""

class RegressionMetrics:
    def  __init__(self):
        pass

    def mse(self, y_true, y_pred):
        """
        :param y_true: array
        :param y_pred: array
        :return: loss
        Sensitive to outliers
        Smooth, differentiable
        """
        return np.mean((y_true - y_pred) ** 2)

    def rmse(self, y_true, y_pred):
        return np.sqrt(self.mse(y_true, y_pred))

    def mae(self, y_true, y_pred):
        """
        Mean absolute error
        :param y_true:
        :param y_pred:
        :return:
        Robust but not smooth
        Non differentiable at 0
        """
        return np.mean(np.abs(y_true - y_pred))

    def huber_loss(self, y_true, y_pred, delta=1.0):
        """
        Huber Loss(Smooth L1)
        :param y_true:
        :param y_pred:
        :return:
        Combine the stability of MSE for small err with,
         robustness of MAE for outliers

        Very robust
        """
        error = y_true - y_pred
        is_small_err = np.abs(error) <= delta
        rmse = 0.5 * (error ** 2)
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        return np.mean(np.where(is_small_err, rmse, linear_loss))

