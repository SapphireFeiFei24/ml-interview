import numpy as np

class Node:
    def __init__(self, feature=None, category=None, threshold=None, left=None, right=None, value = None):
        self.feature = feature  # index of the feature to split on
        self.threshold = threshold
        self.left = left  # left child node
        self.right = right  # right child node
        self.value = value  # leaf value
        self.category = category

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    """ A decision tress classifier based on gini index
    Supports both category and numerical features
    Supports early stop based on min_sample_splits, min_impurity decrease
    """
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


class DecisionTreeRegressor:
    """ A decision tree regressor based on mse
    Support numerical features only.
    Early stop based on max_depth, min_samples, mse changes
    """
    def __init__(self, max_depth=None, min_samples=2, min_mse_changes=1e-7):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_mse_changes = min_mse_changes
        self.root = None  # tree

    def _mse(self, labels):
        """
        Calculate mse for a leaf node
        :param labels: (n_samples, )
        :return: mean squared error
        """
        mean = np.mean(labels)
        return np.mean((labels - mean) ** 2)

    def _mse_gain(self, parent_y, left_y, right_y):
        """
        Calculate the mse gain if split on thres compared to no split
        :param parent_y: (n_samples,)
        :param left_y: left children
        :param right_y: right children
        :return:
        """
        if len(left_y) == 0 or len(right_y) == 0:
            raise ValueError("Children size not valid")
        origin_mse = self._mse(parent_y)
        left_mse = self._mse(left_y)
        right_mse = self._mse(right_y)
        left_w = len(left_y) / len(parent_y)
        right_w = len(right_y) / len(parent_y)
        return origin_mse - (left_w * left_mse + right_w * right_mse)

    def _best_split(self, X, y):
        """
        Find the best feature to split on
        :param X: (n_samples, n_features)
        :param y: (n_samples,)
        :return: (feature_idx, threshold, mse_gain)
        """
        n_samples, n_features = X.shape

        largest_mse_gain = 0
        best_feature, best_split = None, None
        # loop through all features
        for feature in range(n_features):
            values = X[:, feature]
            unique_vals = np.unique(values)

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
            for thres in thresholds:
                left_idx = (values <= thres)
                right_idx = ~left_idx
                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    # not a valid split
                    continue

                left_y, right_y = y[left_idx], y[right_idx]
                mse_gain = self._mse_gain(y, left_y, right_y)
                if mse_gain > largest_mse_gain:
                    largest_mse_gain = mse_gain
                    best_feature = feature
                    best_split = thres
        return (largest_mse_gain, best_feature, best_split)

    def _grow_tree(self, X, y, depth=0):
        """
        Grow a tree from node
        :param X: (n_samples, n_features) leftover data within current node
        :param y: (n_samples, ) labels
        :param depth: current depth for this node
        :return: current tree node
        """
        if self.max_depth and depth >= self.max_depth:
            return Node(value=np.mean(y))

        if len(y) < self.min_samples:
            return Node(value=np.mean(y))

        gain, feature, thres = self._best_split(X, y)
        if gain < self.min_mse_changes:
            return Node(value=np.mean(y))

        left_idx = (X[:, feature] <= thres)
        right_idx = ~left_idx

        if left_idx.sum() == 0 or right_idx.sum() == 0:
            return Node(value=np.mean(y))

        left_X, right_X = X[left_idx], X[right_idx]
        left_y, right_y = y[left_idx], y[right_idx]
        left_child = self._grow_tree(left_X, left_y, depth+1)
        right_child = self._grow_tree(right_X, right_y, depth+1)
        node = Node(feature=feature, threshold=thres, left=left_child, right=right_child)
        return node

    def fit(self, X, y):
        """
        Grow a tree based on the training data
        :param X: (n_samples, n_features)
        :param y: (n_samples,)
        :return:
        """
        self.root = self._grow_tree(X, y)

    def _predict_one(self, x):
        """
        :param x: (1, n_features)
        :return: value
        """
        node = self.root
        while not node.is_leaf():
            feat = node.feature
            if x[feat] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        """
        Making predictions
        :param X: (n_samples, n_features)
        :return: predictions (n_samples,)
        """
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)

def test_regression_tree():
    # Small dataset
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5]
    ])
    y = np.array([1.1, 1.9, 3.0, 4.1, 4.9])

    # Train the decision tree
    tree = DecisionTreeRegressor(max_depth=2, min_samples=1)
    tree.fit(X, y)

    # Predict
    preds = tree.predict(X)

    print("Predictions:", preds)
    print("True labels:", y)

    # Simple assertion: predictions are close to original labels
    assert np.allclose(preds, y, atol=1.0), "Predictions not close to labels!"
    print("Test passed ✅")

def test_classifier():
    X = np.array([
        [2.5, 'red'],
        [1.0, 'blue'],
        [3.1, 'red'],
        [2.0, 'green'],
        [2.2, 'blue']
    ], dtype=object)

    y = np.array([0, 0, 1, 1, 0])

    tree = DecisionTreeClassifier(
        max_depth=3,
        category_features=[1]  # feature #1 is categorical
    )
    tree.fit(X, y)
    print(tree.predict([
        [2.4, 'red'],
        [1.5, 'blue'],
        [2.1, 'green']
    ]))
if __name__ == "__main__":
    test_regression_tree()


