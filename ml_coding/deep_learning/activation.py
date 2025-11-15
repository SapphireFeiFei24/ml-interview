import numpy as np

class ActivationFuncs:
    def __init__(self):
        pass

    def sigmoid(self, x):
        """
        Numeric stable version. For large negative x it can overflow
        :param X:
        :return:
        """
        out = np.zero_list(x, dtype=float)
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        # for x > 0
        out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))

        # for x < 0 -> rewrite as exp(x) / (1 + exp(x))
        exp_x = np.exp(x[neg_mask])
        out[neg_mask] = exp_x / (1 + exp_x)

        return out

    def sigmoid_grad(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x):
        """
        (exp(x) - exp(-x)) / (exp(x) + (exp(-x))
        Naive version will cause stack overflow when x is a very large pos/neg number
        :param X:
        :return:
        """
        out = np.zeros_like(x)
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        # for x >= 0, use exp(-2x)
        exp_neg_2x = np.exp(-2 * x[pos_mask])
        out[pos_mask] = 1 - 2 / (1 + exp_neg_2x)

        # for x <= 0, use exp(2x)
        exp_2x = np.exp(2 * x[neg_mask])
        out[neg_mask] = 2 / (1 + exp_2x) - 1
        return out

    def tanh_grad(self, x):
        t = np.tanh(x)
        return 1 - t ** 2

    def relu(self, x):
        np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_grad(self, x, alpha=0.01):
        grad = np.ones_like(x)
        grad[x < 0] = alpha
        return grad

    def softmax(self, x):
        """
        Stable ver. Subtract the max before exponentiating
        :param x:
        :return:
        """
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)

    def softmax_grad(self, x):
        """
        i = k: si(1-si)
        i != k: -sisk
        J = diag(s) - ss^T
        :param X:
        :return:
        """
        s = self.softmax(x)
        return np.diag(s) - np.outer(s, s)