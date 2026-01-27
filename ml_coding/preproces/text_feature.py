"""
Text based feature
* Encode with pretrained embedding
* Convert into raw features: wordcount, category
* Convert into score features: using model
"""
import pandas as pd


def get_word_count(record: str) -> int:
    return len(record)

def get_word_count(record: pd.Series) -> int:
    return record.str.len()