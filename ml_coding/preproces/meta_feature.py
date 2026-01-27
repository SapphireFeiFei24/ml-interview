"""
1. Identifiers
user_id, item_id, author_id
Don't: normalize
Do: Join, Hash/Embeddings

2. Meta data
language, location, age, client_type
Properties: Stale, Missing

3. Time
event_ts, item_created_ts
use relevant time to model ages
use sin/cosine to model cyclical/circle(time of the day, day of the week, month of the year)
use explicit signal for holidays(is_holiday, is_weekend)
"""
import math
import numpy as np

def hours_between(event_time, created_time):
    return (event_time - created_time).total_seconds / 3600

def post_age(curr_time, created_time):
    """
    Should always be relative, never absolute
    :param curr_time:
    :param created_time:
    :return:
    """
    return curr_time - created_time

def time_decay(post_age, half_life):
    """
    Capture freshness, can use model learn itself
    :param post_age:
    :param half_life:
    :return:
    """
    return math.exp(-post_age/half_life)

def cyclic_time(time, base=24):
    """
    Node: Need both sine and cos to ensure it's unique
    :param time: hour of the day, time of the week, month of the year
    :param base: largest duration for the unit eg hour=24
    :return: sin, cos
    """
    two_pi = 2 * np.pi
    return np.sin(two_pi * time / base), np.cos(two_pi * time / base)