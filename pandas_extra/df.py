def reset_multicolumn_index(dataframe):
    columns = ['_'.join(col).strip("_") for col in dataframe.columns.values]
    dataframe.columns = columns