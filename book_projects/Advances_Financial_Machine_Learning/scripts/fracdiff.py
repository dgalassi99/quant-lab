import numpy as np
import pandas as pd

### SNIPPET 5.1 -----------------------------------------------------------------------#

def getWeights(d,size):
  '''
  d = frac diff order
  size = how many weights to return (basically the maximum lag)
  '''
  # thres>0 drops insignificant weights
  w=[1.]
  for k in range(1,size):
    w_ = -w[-1]/k*(d-k+1)
    w.append(w_)
  w = np.array(w[::-1]).reshape(-1,1)
  return w


### SNIPPET 5.2 -----------------------------------------------------------------------#

def getWeights_FFD(d, thres=1e-5):
    """
    Compute fractional differencing weights with thresholding (FFD).

    Parameters
    ----------
    d : float
        Fractional differencing order.
    thres : float
        Minimum absolute weight to keep.

    Returns
    -------
    w : np.array
        Fractional differencing weights (column vector), truncated to keep only significant weights.
    """
    w = [1.0]  # weight for lag 0
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)  # recursive formula
        if abs(w_) < thres:            # stop if weight is too small
            break
        w.append(w_)
        k += 1

    w = np.array(w[::-1]).reshape(-1, 1)  # reverse so lag 0 is last
    return w

### SNIPPET 5.3 -----------------------------------------------------------------------#

def fracDiff(series,d,threshold=.01):
  '''
  Increasing width window, with treatment of NaNs
  Note 1: For thres=1, nothing is skipped.
  Note 2: d can be any positive fractional, not necessarily bounded [0,1].
  '''
  #compute weights for the longest series
  w = getWeights(d,series.shape[0])
  #determine initial calcs to be skipped based on weight-loss threshold
  w_= np.cumsum(abs(w))
  w_ /= w_[-1]
  #num of obs to drop to keep weight loss below threshold
  skip = w_[w_>threshold].shape[0]
  #apply fracdiff
  df = {}
  for name in series.columns:
    seriesF, df_ = series[[name]].ffill().dropna(),pd.Series()
    for iloc in range(skip,seriesF.shape[0]):
      loc = seriesF.index[iloc]
      if not np.isfinite(series.loc[loc,name]):
        continue # exclude NAs
      df_[loc] = np.dot(w[-(iloc+1):,:].T,seriesF.loc[:loc])[0,0]
    df[name] = df_.copy(deep=True)
  df = pd.concat(df,axis=1)
  return df

### SNIPPET 5.4 -----------------------------------------------------------------------#

def fracDiff_FFD(series,d,thres=1e-5):
    # 1) Compute weights for the longest series
    w = getWeights_FFD(d, thres)   # truncated weights, length = width+1
    width = len(w) - 1             # number of lags kept

    # 2) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].ffill().dropna(), pd.Series()

        # loop over available windows
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1-width], seriesF.index[iloc1]

            if not np.isfinite(series.loc[loc1, name]):
                continue # skip NaNs

            # dot product: weights ⋅ lagged values
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0,0]

        df[name] = df_.copy(deep=True)

    df = pd.concat(df, axis=1)
    return df