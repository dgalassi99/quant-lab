import numpy as np
import pandas as pd

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



######## MODIFIED VERSION OF getTBMLabels ###
def getTBMLabels_withVB(events, close):
    # 1) Align prices with event start and end times (t1)
    events_ = events.dropna(subset=['t1'])  # keep events that have an end time
    px = events_.index.union(events_['t1'].values).drop_duplicates()  # union of start and end times
    px = close.reindex(px, method='bfill')  # get prices at these times, backfill missing

    # 2) Create output DataFrame indexed by event start times
    out = pd.DataFrame(index=events_.index)

    # Calculate return between event start and end time
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1

    # Assign bins based on return sign (+1, 0, -1)
    out['bin'] = np.sign(out['ret'])
#### NEW MODIFICATION ####
    #override with 0 when vertical barrier was first hit
    out.loc[events_['hit_first'] == 'vb', 'bin'] = 0
###########################
    out['t1'] = events_['t1']  # include event end time

    return out


### MODIFIED VERSION OF getEvents ###

def getEvents_with_hit_type(close, tEvents, ptSl, trgt, minRet, t1 = False):
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

####### MOD #######
    # get the first hit type (tp, sl, or t1)
    # df0 is a DataFrame with columns 'tp', 'sl', and 't
    first_hits = df0.dropna(how='all').idxmin(axis=1)  # returns 'tp', 'sl', or 't1'
    events['hit_first'] = first_hits.map({'tp': 'tp', 'sl': 'sl', 't1': 'vb'})
#####################

    #update t1 wiuth the first event occurring between tp and sl and t1
    events['t1'] = df0.dropna(how='all').min(axis=1)
    
    # remove side column
    events = events.drop('side', axis=1)

    return events

# --- GET THE TARGET RETURN FOR THE TRIPLE BARRIER METHOD --- #

def GetTargetforTBM(close, ema_periods):
    """
    close: pd.Series di prezzi indicizzati da timestamp arbitrari (anche intraday)
    span0: span per la media mobile esponenziale della volatilità
    Restituisce: pd.Series della volatilità stimata (deviazione std dei ritorni log)
    """

    # calcola i ritorni logaritmici
    log_ret = np.log(close).diff()

    # stima la volatilità come deviazione standard mobile esponenziale dei ritorni logaritmici
    vol = log_ret.ewm(span=ema_periods).std()
    # fill teh starting Nan created by the emw
    vol = vol.bfill()

    return vol