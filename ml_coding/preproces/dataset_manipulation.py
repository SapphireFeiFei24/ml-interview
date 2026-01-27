"""
Checklist before start
1. Input Data: Data size, duplication, known corruption
2. Label Distribution: distribution, metrics to use later
3. Missing features
4. Feature Distribution: Heavy tail, Outliers, Zero inflation
5. Correlations: Feature -> Label, any leakage
"""
import numpy as np
import numpy.random
import pandas

TRAINING_DATA_PATH = "../../data_sample/train.csv"

# 1. Reading from a csv file
import csv
import pandas as pd
def load_data_to_dict(path: str):
    data = []
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(type(row),row)
            data.append(row)
    return data

def load_data_to_dataframe(path: str):
    df = pd.read_csv(path)
    print(df.info())
    return df


# 2. Explore
# 2.1 Data size: filter out invalid if any
def data_size(data):
    return len(data)

def data_size_df(df):
    return df.shape[0]


# 2.1.1 Uniqueness:
## Goal: how dense is the interaction matrix, whether to aggregate over id
def unique_user_size(data: list, id_key):
    return len({row[id_key] for row in data})


def unique_user_df(dataframe: pandas.DataFrame, id_key):
    return dataframe[id_key].nunique


# 2.2 Label Analysis
## Goal: check if any imbalance in training data, might need resampling

def label_distribution(data: list, key: str):
    counter = Counter(row[key] for row in data)
    return counter


def label_dist_df(data: pd.DataFrame, key: str):
    return data[key].value_counts(normalize=True)


# 2.2.1 Resampling if needed
def downsample(data: list, ratio=1.0):
    np_array = np.array(data)
    count = int(np_array.shape[0] * ratio)
    index = np.random.choice(np_array.shape[0], count, replace=False)
    return np_array[index]


def downsample_df(data: pd.DataFrame, ratio=1.0):
    return data.sample(frac=ratio, replace=False)


def upsample(data: list, ratio=1.0):
    np_array = np.array(data)
    count = int(np_array.shape[0] * ratio)
    index = np.random.choice(np_array.shape[0], count, replace=True)
    return np_array[index]


def upsample_df(data: pd.DataFrame, ratio=1.0):
    return data.sample(frac=ratio, replace=True)


# 3. Features
## 3.1 Missing Features
def missing_rate(data: list, key: str):
    return sum(1 if row[key] is None else 0 for row in data) / len(data)


def missing_rate(data: pd.DataFrame, key: str):
    return data[key].isnull().mean()


## 3.2 Distribution of count features
### Identify heavy tail feautres by checking p99/p90 ratio(>=10) etc
def percentile_distribution(data: list, key: str):
    values = sorted(row[key] for row in data if row[key] is not None)
    p90 = values[int(len(values) * 0.9)]
    p99 = values[int(len(values) * 0.99)]
    return p90, p99

def percentile_distribution_df(data: pd.DataFrame, key: str):
    data[key].describe(percentiles=[0.9, 0.99])


## 3.3 High Cardinality Categorical Features
### "If dimension is too high, might need bucketing"

from collections import Counter
## 3.4 Time Consistency
### "Filter out the invalid rows, create time can't be later than ranking time"

# 4. Training & Testing Split
def split_training_testing(input_data):
    """
    Split based on 80-20
    :param input_data:
    :return: training, testing
    """
    x = numpy.array(input_data)
    numpy.random.shuffle(x)
    cnt = len(x)
    return x[:int(cnt*0.8)], x[int(cnt*0.8):]

if __name__=="__main__":
    load_data_to_dataframe(TRAINING_DATA_PATH)









