import pandas as pd

# to remove trend
# create a differenced series
def difference(ts, interval=1):
    diff = list()
    for i in range(interval, len(ts)):
        value = ts[i] - ts[i - interval]
        diff.append(value)
    return pd.Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob