import pandas as pd
import numpy as np

def to_time_format(data,feature_name,date_format):
    return pd.to_datetime(data.loc[:,feature_name],format = date_format,infer_datetime_format = True)


def dummy_time_feature(data,feature_name):
    index = 0
    def func(date,tmp_array):
        nonlocal index
        # tm_year,tm_mon,tm_mday,tm_hour,tm_min,tm_sec,tm_wday,tm_yday,tm_isdst
        tm_year,tm_mon,tm_mday,tm_hour,tm_min,tm_sec,tm_wday,tm_yday,tm_isdst =  date.timetuple()
        tmp_array[index,:] = [tm_year,tm_mon,tm_mday,tm_hour,tm_min,tm_sec,tm_wday,tm_yday,tm_isdst]
        index += 1

    tmp_array = np.empty((data.shape[0],9),dtype = np.int32)
    data.loc[:,"date"].apply(func, args = (tmp_array,))
    return pd.DataFrame(data = tmp_array, index = range(len(tmp_array)), columns = ["tm_year","tm_mon","tm_mday","tm_hour","tm_min","tm_sec","tm_wday","tm_yday","tm_isdst"])
