### COLLECTION OF BOOK SCRIPTS 

# SNIPPET 2.1 -----------------------------------------------------------------------#

import numpy as np

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