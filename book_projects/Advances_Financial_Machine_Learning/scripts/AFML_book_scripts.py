### COLLECTION OF BOOK SCRIPTS 

import numpy as np
import pandas as pd

# SNIPPET 2.1 -----------------------------------------------------------------------#


def pca_weights(cov_matrix, risk_distribution = None, risk_target = 1.0):
    """ 
    Given a cov matrix NxN computes the portfolio weights based on PCA
    risk decomposition

    *Args:
    - cov_matrix --> covariance matrix NxN
    - risk_distribution --> desired risk distribution across components (Nx1 and sums to 1)
    - risk_target --> total portfolio risk
    
    *Returns:
    - weights --> vector of assets weights (Nx1) 
    """
    #eigen decomposition
    eval, evec = np.linalg.eigh(cov_matrix) 
    #sort eval descending 
    indices = eval.argsort()[::-1]
    eval, evec = eval[indices], evec[:, indices]
    
    #if no risk_distr is given assign all risk to the last component
    if risk_distribution is None:
        risk_distribution = np.zeros(cov_matrix.shape[0])
        risk_distribution[-1] = 1

    loads = risk_target*np.sqrt(risk_distribution/eval)
    weights = np.dot(evec, loads.reshape(-1,1))

    return weights

# SNIPPET 2.2 -----------------------------------------------------------------------#

def roll_gaps(series, dictio={'Instrument':'contract_id','Open':'open', 'Close':'close'}, matchEnd=True):
    """
    *Args
    series: dataframe with futures indexed by time
    diction: dict mapping logical names (Instrument) to column names (contract_id,...)
    matchEnd: boolean flag controlling alignment of the cumulative gaps
    """

    #identify roll dates when ticker changes by keeping only first occurence and getting the index
    roll_dates = series[dictio['Instrument']].drop_duplicates(keep='first').index
    #create gaps series with same index as original series initialized as zeros
    #this will have zeros other than when gaps occur
    gaps = series[dictio['Close']]*0
    #convert index (timestamps) of the df into a list
    all_indices = list(series.index)
    #for each roll date find the index of the bar preedicng the roll date 
    prev_bar_positions = [all_indices.index(i) - 1 for i in roll_dates]
    #for all roll dates (except the first) calculate the gap as open(i)-close(i-1)
    gaps.loc[roll_dates[1:]] = series[dictio['Open']].loc[roll_dates[1:]] - series[dictio['Close']].iloc[prev_bar_positions[1:]].values
    #compute cumulative sum of gasp to have th etoal roll adjustment over time
    gaps = gaps.cumsum()
    #if true sub last gap value so the series ends at zero 
    #this aligns the rolled series to have the same price level as the raw series
    if matchEnd:
        gaps -= gaps.iloc[-1]
    
    return gaps
    

def get_rolled_series_from_df(df, matchEnd=True):
    """
    Apply roll gap adjustment to a DataFrame with synthetic futures data.
    
    Parameters:
    - df: DataFrame with 'contract_id', 'open', and 'close' columns.
    - matchEnd: Whether to roll backward (True) or forward (False).
    
    Returns:
    - df: Modified DataFrame with 'Close_adj' column.
    """
    df = df.copy()
    #get the roll gaps from roll_gaps function
    gaps = roll_gaps(df, matchEnd=matchEnd)
    #adjust close price by subtracting gaps
    for fld in ['close']:
        df[f'{fld}_adj'] = df[fld] - gaps
    return df

# SNIPPET 2.2 - EXTRA TO GEN SYNTHETIC FUTURE DATA -----------------------------------------------------------------------#

def generate_synthetic_futures_data():
    bars_per_contract = 30  # e.g. 12 bars per contract (you can increase if needed)
    num_contracts = 20
    total_bars = bars_per_contract * num_contracts

    time_index = pd.date_range(start='2025-01-01', periods=total_bars, freq='D')
    tickers = [f"FUT_{i:02d}" for i in range(1, num_contracts + 1) for _ in range(bars_per_contract)]

    # Simulate realistic-looking price series with gaps between rolls
    prices = []
    open_prices = []

    last_close = 100
    for i in range(num_contracts):
        contract_returns = np.random.normal(0, 0.5, bars_per_contract)
        contract_prices = last_close + np.cumsum(contract_returns)
        prices.extend(contract_prices)
        open_prices.extend([contract_prices[0]] + contract_prices[:-1].tolist())
        # Add a roll gap between contracts
        last_close = contract_prices[-1] + np.random.normal(1.0, 0.5)

    df = pd.DataFrame({
        'date': time_index,
        'contract_id': tickers,
        'open': open_prices,
        'close': prices,
    })
    df = df.set_index('date')
    return df

# SNIPPET 2.3 - COMPUTE NON-NEGATIVE PRICE INDEX SERIES -----------------------------------------------------------------------#

def compute_non_negative_rolled_price_index(df, 
                                dictio={'Instrument': 'contract_id', 'Open': 'open', 'Close': 'close'}, 
                                matchEnd=True):
    """
    Compute a non-negative rolled price series (simulated $1 investment)
    from a raw futures price DataFrame.

    Parameters:
    - df: DataFrame containing futures price data.
    - dictio: Dictionary mapping logical names to column names.
    - matchEnd: If True, align rolled series to end of raw series (backward roll).

    Returns:
    - rolled_df: DataFrame with adjusted close prices, returns, and rolled investment price.
    """

    df = df.copy()
    
    # Step 1: Compute cumulative roll gaps
    gaps = roll_gaps(df, dictio=dictio, matchEnd=matchEnd)

    # Step 2: Adjust the prices by subtracting cumulative roll gaps
    for fld in [dictio['Open'], dictio['Close']]:
        df[f'{fld}_adj'] = df[fld] - gaps

    # Step 3: Compute returns using adjusted close but dividing by raw close
    df['Returns'] = df[f"{dictio['Close']}_adj"].diff() / df[dictio['Close']].shift(1)

    # Step 4: Compute non-negative price index from returns
    df['Price_Index'] = (1 + df['Returns']).cumprod()

    return df

# SNIPPET 2.4 - THE SYMMETRIC CUSUM FILTER -----------------------------------------------------------------------#

def get_CUSUM_events(gRaw, h):
    """
    Symmetric CUSUM filter to detect events in a time series.
    
    Parameters
    ----------
    gRaw : pd.Series
        Raw time series (e.g., close prices).
    h : float or pd.Series
        Threshold to signal a new event.
        Can be a scalar or a time-indexed Series.
    
    Returns
    -------
    pd.DatetimeIndex
        Event timestamps where the filter was triggered.
    """
    tEvents = []
    sPos, sNeg = 0, 0
    diff = gRaw.diff()

    is_series = isinstance(h, (pd.Series, np.ndarray))

    for idx in range(1, len(diff)):  # start from second value
        h_i = h.iloc[idx] if is_series else h
        val = diff.iloc[idx]

        sPos = max(0, sPos + val)
        sNeg = min(0, sNeg + val)

        if sNeg < -h_i:
            sNeg = 0
            tEvents.append(diff.index[idx])
        elif sPos > h_i:
            sPos = 0
            tEvents.append(diff.index[idx])

    return pd.DatetimeIndex(tEvents)


### SNIPPET 3.1 - GET DAILY VOLATILITY ESTIMATES -----------------------------------------------------------------------#

def GetDailyVol(close, span0=100):
  # find the index position of the one day before each date -> find where price was one BAR ago
  df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
  # eliminate if we could not find a prior day
  df0 = df0[df0>0]
  # create a series where values are the previous valid date and the index is aligned to the original dates
  df0 = pd.Series(close.index[df0-1], index=close.index[close.shape[0] - df0.shape[0]:])
  # daily returns between these timestamps
  daily_returns = close.loc[df0.index].values / close.loc[df0.values].values - 1
  daily_returns = pd.Series(daily_returns, index=df0.index)
  # exponentially weighted moving std dev of these returns
  vol = daily_returns.ewm(span=span0).std()
  return vol


### SNIPPET 3.2 - TRIPLE BARRIER METHOD -----------------------------------------------------------------------#

def applyTPSLOnT1(close, events, tpsl, molecule):
    """
    close = pd Series of prices indexed by datetime
    events = DataFrame with at least:
      - t1: timestamp when event ends (vertical barrier) 
      - trgt: target return for TP/SL --> can be a constant or a for example a volatility estimate
      - side: -1/1 (short/long)
    tpsl = (tp, sl) tuple, e.g. (1,1) for postioning multiplier (1,1) means symmetric TP/SL
    molecule = subset of indices (for parallelization) --> you can slice and then concatenate later 
    
    Note:
    molecule ---> AVOID ID NOW pass all events without parallelization 
    """
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)

    # profit taking threshold
    if tpsl[0] > 0: #could be also
        tp = tpsl[0] * events_['trgt'] #all trgt (as tp is 1 or 0 without tp)
    else:
        tp = pd.Series(index=events_.index)  # all NaNs

    # stop loss threshold (same logic as tp)
    if tpsl[1] > 0:
        sl = -tpsl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events_.index)
    #loop in the events after filling the last rows due to NaN

    #loc is the start time
    #t1 is the vertical barrier ti,e
    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1] #prices between loc and t1
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side'] #calculate returns and multiply by side (short or long)
        #df0[df0 < sl[loc]] gives all timestamps where returns fell below the SL level.
        #.index.min() gets the earliest timestamp where this happened.
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        #same logic applies here....
        out.loc[loc, 'tp'] = df0[df0 > tp[loc]].index.min()

    return out

### SNIPPET 3.3 - GET EVENTS-----------------------------------------------------------------------#

def getEvents(close, tEvents, ptSl, trgt, minRet, t1 = False):
    """
    close: pd.Series or prices indexed by datetime
    tEvents: timeindex with timestamps of events selected by a sampling procedure (ex CUSUM)
    ptSl: tuple (tp, sl) take profit e stop loss multipliers (for eventual asymetr)c positioning)
    trgt: pd.Series of targets expresses in term of abs returns (e.g. 0.02 for 2% target)
    minRet: min target return to consider an event
    t1: pd.Series optional for vertical barriers (max holding period), default False (no limit)
    
    retunrs...

    df with:
    - t1: timestamp at which the first barrier is hit (either tp, sl or vertical barrier)
    - trgt: target return for the event used to generate the barriers 
    
    Note:
    - to increase th enumebr of t1 = veticla barriers we need to increaet he multipliets tpsl
    - increasing minret will only filter out more events
    """

    # filter for only tEvents and consider only those above minRet
    trgt_subset = trgt.loc[tEvents]
    trgt_subset = trgt_subset[trgt_subset > minRet]

    #get the vertical barrier
    if t1 is False:
        t1= pd.Series(pd.NaT, index=trgt_subset.index)

    #set side (here always long = 1)
    side_ = pd.Series(1., index=trgt_subset.index)

    # events dataframe
    events = pd.concat({'t1': t1, 'trgt': trgt_subset, 'side': side_}, axis=1).dropna(subset=['trgt'])

    # applyTPSLOnT1 (no multiprocessing)
    df0 = applyTPSLOnT1(close=close, events=events, tpsl=ptSl, molecule=events.index)

    #update t1 wiuth the first event occurring between tp and sl and t1
    events['t1'] = df0.dropna(how='all').min(axis=1)
    
    # remove side column
    events = events.drop('side', axis=1)

    return events

### SNIPPET 3.5 - TRIPLE BARRIER METHOD LABELLING-----------------------------------------------------------------------#

def getTBMLabels(events, close):
    """
    Compute meta-labels for trend-following events.

    events: DataFrame with
        - t1: event end timestamp (vertical barrier)
        - trgt: target return
        - side (optional): side of the trade (-1/1)
    close: pd.Series of prices indexed by datetime

    Returns:
        DataFrame indexed by event start time with:
        - ret: return during event (multiplied by side if present)
        - bin: label (0/1 for meta-labeling, -1/1 if no side)
        - t1: event end timestamp
    """
    # 1) Align prices with event start and end times (t1)
    events_ = events.dropna(subset=['t1'])  # keep events that have an end time
    px = events_.index.union(events_['t1'].values).drop_duplicates()  # union of start and end times
    px = close.reindex(px, method='bfill')  # get prices at these times, backfill missing

    # 3) Create output DataFrame
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1

    # 4) Meta-labeling: multiply by side if it exists
    if 'side' in events_:
        out['ret'] *= events_['side']
        out['bin'] = np.sign(out['ret'])
        out.loc[out['ret'] <= 0, 'bin'] = 0  # unprofitable → 0
    else:
        out['bin'] = np.sign(out['ret'])  # -1/+1

    # 5) Include t1
    out['t1'] = events_['t1']
    out['hit_first'] = events_['hit_first']

    return out


### SNIPPET 3.6 - GetEvents for METALABELLING-----------------------------------------------------------------------#

def getEventsMeta(close, tEvents, ptSl, trgt, minRet, t1= False, side=None):
    """
    close: pd.Series or prices indexed by datetime
    tEvents: timeindex with timestamps of events selected by a sampling procedure (ex CUSUM)
    ptSl: tuple (tp, sl) take profit e stop loss multipliers (for eventual asymetr)c positioning)
    trgt: pd.Series of targets expresses in term of abs returns (e.g. 0.02 for 2% target)
    minRet: min target return to consider an event
    t1: pd.Series optional for vertical barriers (max holding period), default False (no limit)
    side: pd.Series with the side of the trade (1 for long, -1 for short), if None it assumes adirectional 
    
    retunrs...

    df with:
    - t1: timestamp at which the first barrier is hit (either tp, sl or vertical barrier)
    - trgt: target return for the event used to generate the barriers 
    
    Note:
    - to increase th enumebr of t1 = veticla barriers we need to increaet he multipliets tpsl
    - increasing minret will only filter out more events
    """

    # filter for only tEvents and consider only those above minRet
    trgt_subset = trgt.loc[tEvents]
    trgt_subset = trgt_subset[trgt_subset > minRet]

    #get the vertical barrier
    if t1 is False:
        t1= pd.Series(pd.NaT, index=trgt_subset.index)


    if side is None:
        side_ = pd.Series(1., index=trgt.index)
        ptSl_ = [ptSl[0], ptSl[0]]  # symmetric TP and SL
    else:
        side_ = side.loc[trgt.index]
        ptSl_ = ptSl[:2]  # assume first two elements correspond to TP and SL multipliers 
    
    # events dataframe
    events = pd.concat({'t1': t1, 'trgt': trgt_subset, 'side': side_}, axis=1).dropna(subset=['trgt'])

    # applyTPSLOnT1 (no multiprocessing)
    df0 = applyTPSLOnT1(close=close, events=events, tpsl=ptSl_, molecule=events.index)
    # save the first hit (tp, sl, or t1) in a new column
    first_hits = df0.dropna(how='all').idxmin(axis=1)  # returns 'tp', 'sl', or 't1'
    events['hit_first'] = first_hits.map({'tp': 'tp', 'sl': 'sl', 't1': 'vb'})
    #update t1 wiuth the first event occurring between tp and sl and t1
    events['t1'] = df0.dropna(how='all').min(axis=1)

    return events

### SNIPPET 3.7 - GetTBMLabels for METALABELLING-----------------------------------------------------------------------#

def getTBMLabelsMeta(events, close):
    """
    Compute meta-labels for trend-following events.

    events: DataFrame with
        - t1: event end timestamp (vertical barrier)
        - trgt: target return
        - side (optional): side of the trade (-1/1)
    close: pd.Series of prices indexed by datetime

    Returns:
        DataFrame indexed by event start time with:
        - ret: return during event (multiplied by side if present)
        - bin: label (0/1 for meta-labeling, -1/1 if no side)
        - t1: event end timestamp
    """
    # 1) Keep events that have an end time
    events_ = events.dropna(subset=['t1'])

    # 2) Align prices with event start and end times
    px_index = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px_index, method='bfill')

    # 3) Create output DataFrame
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1

    # 4) Meta-labeling: multiply by side if it exists
    if 'side' in events_:
        out['ret'] *= events_['side']
        out['bin'] = np.sign(out['ret'])
        out.loc[out['ret'] <= 0, 'bin'] = 0  # unprofitable → 0
    else:
        out['bin'] = np.sign(out['ret'])  # -1/+1

    # 5) Include t1 and hit first
    out['t1'] = events_['t1']
    out['hit_first'] = events_['hit_first']

    return out

### SNIPPET 4.1 - NumCoEvents -----------------------------------------------------------------------#

def mpNumCoEvents(closeIdx, t1, molecule):
    '''
    Compute the number of concurrent events per bar.
    + molecule[0] is the date of the first event on which the weight will be computed
    + molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[molecule].max() impacts the count.
    '''
    t1 = t1.fillna(closeIdx[-1])  # unclosed events are set to the last available bar
    t1 = t1[t1 >= molecule[0]]    # keep events that end at or after molecule[0]
    t1 = t1.loc[:t1[molecule].max()]  # keep events that start before the last molecule event ends

    #t1 is a matrix with index time ti0 and column ti1 --> intrval for each label

    #find the slice of closeIDX spanning from first event start to last event end
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    #initialize counter series
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1]+1])
    #for each event interval [tIn,tOut] incrememtn the counter
    for tIn, tOut in t1.items():
      count.loc[tIn:tOut] += 1
    return count.loc[molecule[0]:t1[molecule].max()]

### SNIPPET 4.2 - sampleTW -----------------------------------------------------------------------#

def sampleTW(t1, numCoEvents, molecule):
    '''
    t1 = series with start times as index and end times as values
    numCoEvents = series giving the num. of concurrent evetns (from mpNumCoEvents)
    molecule = subset of events start times
    '''
    # Derive average uniqueness over the event's lifespan
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (1./numCoEvents.loc[tIn:tOut]).mean()
    return wght

### SNIPPET 4.3 -----------------------------------------------------------------------#

def getIndMatrix(barIx, t1):
    """
    Build the indicator matrix 1_{t,i}.
    Rows = time t, Cols = labels i
    
    Parameters
    ----------
    barIx : pd.Index (DatetimeIndex or RangeIndex)
        Full index of bars (timeline).
    t1 : pd.Series
        Event end times (index = event start time).
    """
    indM = pd.DataFrame(0, index=barIx, columns=range(len(t1)))
    
    for i, (t0, t1_end) in enumerate(t1.items()):
        if isinstance(barIx, pd.DatetimeIndex):
            # works with datetime
            indM.loc[t0:t1_end, i] = 1
        else:
            # works with integer bar indices
            indM.loc[t0:t1_end-1, i] = 1
    
    return indM


### SNIPPET 4.4 -----------------------------------------------------------------------#

def getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix
    c = indM.sum(axis=1)         # concurrency: how many labels are active at each bar
    u = indM.div(c, axis=0)      # uniqueness = 1/concurrency for each active label
    avgU = u[u > 0].mean()       # average uniqueness per column (label)
    return avgU

### SNIPPET 4.5 -----------------------------------------------------------------------#

def seqBootstrap(indM,sLength=None):
  #generate a sample via sequential boostrap
  phi = []
  if sLength is None:
    sLength = indM.shape[1]
  while len(phi) < sLength:
    avgU = pd.Series()
    for i in indM:
      indM_ = indM[phi + [i]]  # keep only already picked labels + candidate i
      avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
    prob = avgU / avgU.sum()
    phi += [np.random.choice(indM.columns, p=prob)] #list of the indices of labels picked
  return phi


### SNIPPET 4.6 -----------------------------------------------------------------------#

def getRndT1(numObs, numBars, maxH):
    """
    Generate a random t1 Series for numObs observations.
    Each observation starts at a random bar and spans a random number of bars.

    Returns a sorted pandas Series: index = start bar (t0), value = end bar (t1)
    """
    t1 = pd.Series(dtype=int)
    for _ in range(numObs):
        ix = np.random.randint(0, numBars)               # random start bar
        val = ix + np.random.randint(1, maxH+1)         # end bar (span at least 1)
        t1.loc[ix] = val
    return t1.sort_index()


### SNIPPET 4.7 -----------------------------------------------------------------------#

def auxMC(numObs, numBars, maxH):
    """
    Monte Carlo trial comparing standard vs sequential bootstrap.

    Parameters:
    - numObs : int : number of observations (I)
    - numBars : int : total number of time bars (T)
    - maxH : int : maximum bar span for each observation

    Returns:
    - dict with 'stdU' (standard bootstrap avg uniqueness)
      and 'seqU' (sequential bootstrap avg uniqueness)
    """

    # Generate random t1 series for each observation
    t1 = getRndT1(numObs, numBars, maxH)

    # Build indicator matrix
    barIx = range(t1.max() + 1)
    indM = getIndMatrix(barIx, t1)

    # Standard bootstrap sampling
    phi_std = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = getAvgUniqueness(indM[phi_std]).mean()

    # Sequential bootstrap sampling
    phi_seq = seqBootstrap(indM)
    seqU = getAvgUniqueness(indM[phi_seq]).mean()

    return {'stdU': stdU, 'seqU': seqU}


### SNIPPET 4.10 -----------------------------------------------------------------------#

def sampleW(t1, numCoEvents, close, molecule):
    """
    Derive sample weights by absolute return attribution.

    Parameters
    ----------
    t1 : pd.Series
        Series with start (index) and end (value) times for each event.
    numCoEvents : pd.Series
        Number of concurrent events at each time step.
    close : pd.Series
        Price series (used to compute log-returns).
    molecule : list or pd.Index
        Subset of events to process (usually all events).

    Returns
    -------
    wght : pd.Series
        Absolute return weights for each event in molecule.
    """
    # log-returns
    ret = np.log(close).diff()

    # initialize weight series
    wght = pd.Series(index=molecule, dtype=float)

    # compute weights
    for tIn, tOut in t1.loc[molecule].items():
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[tIn:tOut]).sum()

    wght = wght.abs()
    wght = wght / wght.sum()

    # return normalized absolute weights
    return wght


### SNIPPET 4.11 -----------------------------------------------------------------------#

def getTimeDecay(tW, clfLastW=1.0):
    """
    Apply piecewise-linear decay to observed uniqueness (tW).

    Parameters
    ----------
    tW : pd.Series
        Observed uniqueness values (by event).
    clfLastW : float, default=1.0
        - If >= 0: newest observation has weight = 1, oldest has weight = clfLastW.
        - If < 0: newest observation has weight = 1, and some older ones decay to 0.

    Returns
    -------
    pd.Series
        Decayed weights aligned with tW.
    """
    # cumulative uniqueness (acts as pseudo-time axis)
    clfW = tW.sort_index().cumsum()

    # determine slope depending on clfLastW regime
    if clfLastW >= 0:
        slope = (1.0 - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1.0 / ((clfLastW + 1) * clfW.iloc[-1])

    # intercept so that newest observation has weight = 1
    const = 1.0 - slope * clfW.iloc[-1]

    # linear decay
    clfW = const + slope * clfW

    # enforce non-negativity
    clfW[clfW < 0] = 0.0

    print(const, slope)  # keep your debug print

    return clfW