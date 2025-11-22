import numpy as np

class Node:
    def __init__(self, feature=None, category=None, threshold=None, left=None, right=None, value = None):
        self.feature = feature  # index of the feature to split on
        self.threshold = threshold
        self.left = left  # left child node
        self.right = right  # right child node
        self.value = value  # leaf value
        self.category=category

    def is_leaf(self):
        return self.value is not None


class CART:
    def __init__(self, max_depth=None, min_samples_split=2,
                 min_impurity_decrease=1e-7, category_features=None):
        self.max_depth = max_depth
        self.min_samples_split=min_samples_split
        self.min_impurity_decrease=min_impurity_decrease
        self.categorical_features = set(category_features or [])
        self.root = None

    def _gini(self, y):
        """
        Calculate gini score within a node
        1 - sum(Ck/C)^2
        :param y: (n_samples,)
        :return: gini score (impurity)
        """
        # Get the counts for each unique ys
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()  # (n_class,)
        return 1 - np.sum(p ** 2)

    def _information_gain_w_baseline(self, baseline_gain, parent_y, left_y, right_y):
        """
        Calculate information gain based on a split
            1 * gini(parent) - (prob(left) * gini(left) + prob(right) * gini(right))
        :param baseline_gain: the impurity of parent_y
        :param parent_y: (n_left+n_right,)
        :param left_y: (n_left,)
        :param right_y: (n_right,)
        :return: information gain with the split of (left, right)
                """
        prob_left = len(left_y) / len(parent_y)
        prob_right = len(right_y) / len(parent_y)
        return baseline_gain - (prob_left * self._gini(left_y) + prob_right * self._gini(right_y))

    def _information_gain(self, parent_y, left_y, right_y):
        """
        Calculate information gain based on a split
          1 * gini(parent) - (prob(left) * gini(left) + prob(right) * gini(right))
        :param parent_y: (n_left+n_right,)
        :param left_y: (n_left,)
        :param right_y: (n_right,)
        :return: information gain with the split of (left, right)
        """
        prob_left = len(left_y) / len(parent_y)
        prob_right = 1 - prob_left
        return self._gini(parent_y) - (prob_left * self._gini(left_y) + prob_right * self._gini(right_y))

    def _best_split(self, X, y):
        """
        Find the best features to split on
        :param X: (n_samples, n_features)
        :param y: (n_samples, )
        :return: best_feature, best_threshold, best_impurity_gain
        """
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            # stop splitting
            return None, 0

        baseline_impurity = self._gini(y)
        best_gain = 0
        best_split = None
        for feature in range(n_features):
            if feature in self.categorical_features:
                # Category split: feature == category
                categories = np.unique(X[:, feature])
                for cat in categories:
                    left_idx = (X[:, feature] == cat)
                    right_idx = ~left_idx
                    if left_idx.sum() == 0 or right_idx.sum() == 0:
                        # All have the same value, continue
                        continue
                    gain = self._information_gain_w_baseline(
                        baseline_impurity, y, y[left_idx], y[right_idx])
                    if gain > best_gain:
                        best_gain = gain
                        best_split = {"feature": feature,
                                      "category": cat,
                                      "threshold": None}
            else:
                # sort by feature for efficient threshold scanning
                idx = np.argsort(X[:, feature])
                X_sorted = X[idx, feature]

                # candidate thresholds = midpoints between unique sorted values
                unique_vals = np.unique(X_sorted)
                if len(unique_vals) == 1:
                    continue

                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2  # (n_uniques - 1,)

                for t in thresholds:
                    left_idx = (X[:, feature] <= t)
                    right_idx = ~left_idx
                    if left_idx.sum() == 0 or right_idx.sum() == 0:
                        continue
                    gain = self._information_gain_w_baseline(baseline_impurity, y, y[left_idx], y[right_idx])
                    if gain > best_gain:
                        best_gain = gain
                        best_split = {
                            "feature": feature,
                            "threshold": t,
                            "category": None
                        }
        return best_split, best_gain

    def _majority_vote(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        if depth >= self.max_depth or \
            n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            # check the condition of not splitting
            return Node(value=self._majority_vote(y))

        split, gain = self._best_split(X, y)
        if split is None or gain < self.min_impurity_decrease:
            return Node(value=self._majority_vote(y))

        feature = split["feature"]
        threshold = split["threshold"]
        category = split["category"]

        # split data
        left_idx, right_idx = None, None
        if feature in self.categorical_features:
            left_idx = (X[:, feature] == category)
        else:
            left_idx = (X[:, feature] <= threshold)
        right_idx = ~left_idx

        left_child = self._grow_tree(X[left_idx], y[left_idx], depth+1)
        right_child = self._grow_tree(X[right_idx], y[right_idx], depth+1)

        return Node(feature=feature, threshold=threshold, category=category,
                    left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _predict_one(self, x, node):
        while not node.is_leaf():
            if node.category is not None:
                if x[node.feature] == node.category:
                    node = node.left
                else:
                    node = node.right
            else:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
        return node.value

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

if __name__ == "__main__":
    X = np.array([
        [2.5, 'red'],
        [1.0, 'blue'],
        [3.1, 'red'],
        [2.0, 'green'],
        [2.2, 'blue']
    ], dtype=object)

    y = np.array([0, 0, 1, 1, 0])

    tree = CART(
        max_depth=3,
        category_features=[1]  # feature #1 is categorical
    )
    tree.fit(X, y)

    print(tree.predict([
        [2.4, 'red'],
        [1.5, 'blue'],
        [2.1, 'green']
    ]))
