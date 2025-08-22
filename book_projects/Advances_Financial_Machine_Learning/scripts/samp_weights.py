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


