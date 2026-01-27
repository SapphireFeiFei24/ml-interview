"""
1. Extract most common features to prevent overfitting
2. Simple One-hot
"""
from collections import Counter

import pandas as pd


def get_most_common_k(data, name, top_k=10):
    counter = Counter(row[name] for row in data if row.get(name))
    return counter.most_common(top_k)

def get_most_common_k(data: pd.Series, top_k=10):
    data.dropna().value_counts().nlargest(top_k).keys().tolist()

def normalize_genre(data, feature_name, top_k):
    """
    Keep only most common categories to prevent from overfitting
    :param data:
    :param feature_name:
    :param top_k:
    :return:
    """
    top_labels = get_most_common_k(data, feature_name, top_k)
    return [data[feature_name] if feature_name in top_labels else "UNK"]

def normalize_genre(data: pd.Series):
    most_common = get_most_common_k(data)
    def mapping(x):
        if pd.isna(x):
            return ["UNK"]
        return x if x in most_common else "UNK"

    return data.apply(mapping)

def one_hot_encoding(data: pd.Series):
    norm_data = normalize_genre(data)
    norm_data.str.get_dummies().groupby(level=0).max().add_prefix("is_")