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