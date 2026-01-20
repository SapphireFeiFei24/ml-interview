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
"""

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