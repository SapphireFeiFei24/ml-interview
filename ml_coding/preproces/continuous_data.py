"""
Count-based engagement signals
Properties:
Longtailed, Noisy, often partially-observed

Timestamp: never use raw, pay attention to leakage
"""


def hours_between(event_time, created_time):
    return (event_time - created_time).total_seconds / 3600