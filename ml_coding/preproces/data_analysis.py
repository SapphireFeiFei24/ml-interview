"""
Checklist before start
1. Input Data: Data size, duplication, known corruption
2. Label Distribution: distribution, metrics to use later
3. Missing features
4. Feature Distribution: Heavy tail, Outliers, Zero inflation
5. Correlations: Feature -> Label, any leakage
"""
from collections import Counter


def data_size(data):
    return len(data)


def label_distribution(data: dict):
    counter = Counter(row["label"] for row in data)
    return counter

