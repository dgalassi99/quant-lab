### COLLECTION OF SCRIPTS ###

import numpy as np
import pandas as pd

# --- FORM TICK BARS --- #

def TickBarsDf(df,ticks_per_bar):
    '''Takes a df with prices, volume and datetime
    
    Creates a OLHCV type dataset with start and end time
    
    '''
    df = df.copy()
    df['bar_id'] = df.index // ticks_per_bar
    tick_bars_df = df.groupby('bar_id').agg(
        open = ('price','first'),
        low = ('price','min'),
        high = ('price','max'),
        close = ('price','last'),
        volume = ('volume','sum'),
        start_date = ('datetime','first'),
        end_date = ('datetime','last')
    )

    return tick_bars_df

# --- FORM VOLUME BARS --- #

def VolumeBarsDf(df,volume_per_bar):
    bars = [] #list to store the bars
    cum_vol = 0 #counter of cumulative volume starts at 0
    idx = 0 #index starts at 0
    for i, vol in enumerate(df.volume):
        cum_vol += vol #add vol to cumulative vol until...
        if cum_vol >= volume_per_bar: #... the threshold, if reached
            if i+1 > idx:
                bar = df.iloc[idx:i+1] #take the barst between current idx and the moment in which threshold is reached
                bars.append({
                    'open':bar.price.iloc[0],
                    'low':bar.price.min(),
                    'high':bar.price.max(),
                    'close':bar.price.iloc[-1],
                    'volume':bar.volume.sum(),
                    'start_date':bar.datetime.iloc[0],
                    'end_date':bar.datetime.iloc[-1]})
                cum_vol = 0 #reset the cumulative at 0
                idx = i+1 #the new starting index is the index of the last tick of the previous vol bar
    bars_df = pd.DataFrame(bars)
    return bars_df

### USE THE VECTORIZZED VERSION WHEN DF >~ 10e+5 observations ###

def VolumeBarsDfVectorized(df,volume_per_bar):
    df.copy()
    cum_vol = df.volume.cumsum().values #cumulative volume
    n_bars = int(cum_vol[-1]//volume_per_bar) #round down integer of tot vol / vol x bar
    if n_bars == 0:
        print('Not enough cumulative volume to compute even one bar')
    vol_thresholds = volume_per_bar * np.arange(1,n_bars+1) #series fo values multiples of vol_per_bar
    bar_ends = np.searchsorted(cum_vol,vol_thresholds) #finds indices where cum vol goes over the respective threshold
    #first bar and index 0, then each next bar start after the precedent bar ends
    bar_starts = np.concatenate(([0],bar_ends[:-1]+1))  
    bars = []
    for start, end in zip(bar_starts, bar_ends):
            bar = df.iloc[start:end+1]
            bars.append({
                'open': bar['price'].iloc[0],
                'high': bar['price'].max(),
                'low': bar['price'].min(),
                'close': bar['price'].iloc[-1],
                'volume': bar['volume'].sum(),
                'start_date': bar['datetime'].iloc[0],
                'end_date': bar['datetime'].iloc[-1]
            })
        
    return pd.DataFrame(bars)
    
# --- FORM DOLLAR BARS --- #

### DIRECTLY THE VECTORIZZED VERSION  ###

def DollarBarsDfVectorized(df, dollar_per_bar):
    df = df.copy()
    df['dollar_value'] = df['price'] * df['volume']
    cum_dollar = df['dollar_value'].cumsum().values
    n_bars = int(cum_dollar[-1] // dollar_per_bar)
    if n_bars == 0:
        return pd.DataFrame()
    
    thresholds = dollar_per_bar * np.arange(1, n_bars + 1)
    bar_ends = np.searchsorted(cum_dollar, thresholds)
    # Cap bar_ends to the max index available
    bar_ends = np.minimum(bar_ends, len(df) - 1)
    bar_starts = np.concatenate(([0], bar_ends[:-1] + 1))
    bars = []
    for start, end in zip(bar_starts, bar_ends):
        if start > end:
            # Skip invalid bar
            continue
        bar = df.iloc[start:end+1]
        bars.append({
            'open': bar['price'].iloc[0],
            'high': bar['price'].max(),
            'low': bar['price'].min(),
            'close': bar['price'].iloc[-1],
            'volume': bar['volume'].sum(),
            'dollar_volume': bar['dollar_value'].sum(),
            'start_date': bar['datetime'].iloc[0],
            'end_date': bar['datetime'].iloc[-1]
        })
    
    return pd.DataFrame(bars)

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

##### TRIPLE BARRIER METHOD #####

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
