import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic


def difference(ts, interval=1):
    """ remove trend from time series
    
    Parameters
    ----------
    ts (pd.Series) : input time series
    interval (int):  
        
    Returns
    -------
    time series detrend
    """
    diff = list()
    for i in range(interval, len(ts)):
        value = ts[i] - ts[i - interval]
        diff.append(value)
    return pd.Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob



# Stationarity tests
def test_stationarity(ts):
    """ Perform Dickey-Fuller test
    
    Parameters
    ----------
    ts (pd.Series) : input time series
        
    Returns
    -------
    test output
    """
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

