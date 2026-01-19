"""
Example:
    Click count, Like count, Follower Count
1. Log with Cap: prevent gradient explosion while preserve ordering
2. Ratio: Click count/author_follow_count, need to pay attention to zero division
"""

import numpy

def safe_integer(x, default=0):
    """
    Return the original value if positive and is instance of int, else return default value
    :param x:
    :param default:
    :return:
    """
    return x if isinstance(x, int) and x >= 0 else default

def log_cap(x, cap=1000):
    """
    Return value stable log value of x
    :param x:
    :param cap:
    :return:
    """
    return numpy.log1p(min(safe_integer(x), cap))  # log(1 + x)