import numpy as np

# Here are some aggregation functions used in DataFrame.groupby.agg()

unique = lambda x : x.unique().shape[0]

not_zero = lambda x : np.sum(x != 0)