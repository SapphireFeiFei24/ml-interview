"""
1. Extract most common features to prevent overfitting
2. Simple One-hot
"""
from collections import Counter

def get_most_common_k(data, name, top_k=10):
    counter = Counter(row[name]  for row in data if row.get(name))
    return counter.most_common(top_k)


def transform_most_common(data, feature_name, top_k):
    """
    Keep only most common categories to prevent from overfitting
    :param data:
    :param feature_name:
    :param top_k:
    :return:
    """
    top_labels = get_most_common_k(data, feature_name, top_k)
    return [data[feature_name] if feature_name in top_labels else "UNK"]