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
    '''Takes a df with prices, volume and datetime'''
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
    '''Takes a df with prices, volume and datetime'''
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

def DollarBarsDf(df, dollar_per_bar):
    """
    Create dollar bars from tick data using a simple loop-based method.

    Args:
        df (pd.DataFrame): must contain ['price', 'volume', 'datetime']
        dollar_per_bar (float): target dollar value per bar

    Returns:
        pd.DataFrame: dollar bars with OHLC, volume, dollar_volume, start and end dates
    """
    bars = []
    cum_dollar = 0  # cumulative dollar counter
    idx = 0         # starting index of current bar

    df = df.copy()
    df['dollar_value'] = df['price'] * df['volume']

    for i, dv in enumerate(df['dollar_value']):
        cum_dollar += dv

        if cum_dollar >= dollar_per_bar:
            if i + 1 > idx:
                bar = df.iloc[idx:i+1]
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
                cum_dollar = 0  # reset cumulative dollar
                idx = i + 1     # start next bar from the next tick

    return pd.DataFrame(bars)


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