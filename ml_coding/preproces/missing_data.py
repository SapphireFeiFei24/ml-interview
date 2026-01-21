"""
3 types of Data Missing
* Structural Missing: Feature not applicable
* Data Not Collected: Log Gaps
* Data Corrupted: Parse Error
"""
import pandas as pd


# 1. Count Features(likes, follows, replies)
## Non-negative, heavy-tailed, zero is meaningful
## Strategy: fill zero + missing indicator, sanitize negatives

def sanitize_count(data: list, key: str):
    sanitized_row = [row[key] if row[key] is not None
                                 and row[key] >=0 else 0 for row in data]
    is_nan_row = [True if row[key] is not None
                                 and row[key] >=0 else False for row in data]
    return sanitized_row, is_nan_row

def sanitize_count_df(data: pd.DataFrame, key: str):
    data[f"{key}_sanitized"] = data[key].fillna(0).clip(lower=0)
    data[f"{key}_isnan"] = data[key].isna().astype(int)


# 2.Categorical Features(langugate, client_type)
## Often high-candinality, missingess common, longtail of rare values
## Fill missing with UNK, bucket rare values, keep common ones
import categorical_data as cate_data


def sanitize_cate_count(data: list, key: str):
    return cate_data.transform_most_common(data, key, 5)


def sanitize_cate_count_df(data: pd.DataFrame, key: str):
    top_common = cate_data.get_most_common_k(data[key], 5)
    data[f"{key}_sanitized"] = data[key].fillna("UNK")
    data[f"{key}_sanitized"] = data[key].where(data[key].isin(top_common), "UNK")


# 3. Time Features
## Critical for ranking, common source of leakage, Missing or invalid time breaks everything
## Drop invalid rows
def is_valid_time(row: dict):
    return (
        row["post_created_ts"] is not None
        and row["event_ts"] is not None
        and row["post_created_ts"] <= row["event_ts"]
    )


def sanitize_time(data: list):
    data = [r for r in data if is_valid_time(r)]


def sanitize_time(data: pd.DataFrame):
    df = data.dropna(subset=["post_created_ts", "event_ts"])
    df = data[data["post_created_ts"] <= data["event_ts"]]
