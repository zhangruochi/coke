import numpy as np

# Here are some aggregation functions used in DataFrame.groupby.agg()

unique_count = lambda x : x.unique().shape[0]
non_zero_count = lambda x : np.sum(x != 0)
unique_num = lambda x : x.iloc[0]



