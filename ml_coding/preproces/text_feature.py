"""
Text based feature
* Encode with pretrained embedding
* Convert into raw features: wordcount, category
* Convert into score features: using model
* Normalize before cosine similarity
"""
import pandas as pd
import math
import numpy

def get_word_count(record: str) -> int:
    return len(record)

def get_word_count(record: pd.Series) -> int:
    return record.str.len()

def l2_norm(emb: pd.DataFrames):
    """
    X / |X|^2
    :param emb:
    :return:
    """
    norm = numpy.linalg.norm(emb)
    return emb / norm