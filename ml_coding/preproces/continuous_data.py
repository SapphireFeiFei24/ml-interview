"""
* Strictly positive: Price, duration
* Can it be zero?: discount, shipping fee
* Bounded?: ratings
* heavy-tailed or roughly log-normal
* relative difference or absolute difference matter

Standardization: use Z-score when
* Distribution is roughly symmetric
* No heavy tail
* Feature has natural center
"""
import math
import numpy as np
import pandas as pd
def hours_between(event_time, created_time):
    return (event_time - created_time).total_seconds / 3600

def log_trasnform(price, epsilon):
    """
    Compress scale and reflect relative sensitivity
    :param price:
    :param epsilon:
    :return:
    """
    return math.log(price+epsilon)

def price_process_example():
    # Example DataFrame
    df = pd.DataFrame({
        "price": [10, 2000, 0, 15],
        "category": ["A", "A", "B", "B"],
        "user_avg_spend": [50, 100, 20, 30]
    })

    # 1) Sanitize price
    df["price"] = df["price"].clip(lower=0)

    # 2) Cap price per category at p99
    def cap_p99(group):
        cap = group["price"].quantile(0.99)
        group["price_capped"] = np.minimum(group["price"], cap)
        return group

    df = df.groupby("category").apply(cap_p99)

    # 3) Log-transform
    df["price_log"] = np.log1p(df["price_capped"])

    # 4) Percentile bucket per category
    def bucket_percentile(group, bins=[0.2, 0.4, 0.6, 0.8]):
        quantiles = group["price"].quantile(bins).values
        group["price_bucket"] = pd.cut(group["price"], bins=[-np.inf] + list(quantiles) + [np.inf], labels=False)
        return group

    df = df.groupby("category").apply(bucket_percentile)

    # 5) Relative price to user average
    df["price_rel_user"] = df["price"] / df["user_avg_spend"].clip(lower=1)

    # 6) Zero price indicator
    df["is_free"] = (df["price"] == 0).astype(int)
