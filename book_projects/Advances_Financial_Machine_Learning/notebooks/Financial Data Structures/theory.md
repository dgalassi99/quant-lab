# DATA ANALYSIS - FINANCIAL DATA STRUCTURE

## Types of Financial Data

### Fundamental Data

Mostly accounting data reported quarterly. Those have low frequency and can be used in coupling with higher frequency data.

 Typical issues?

1. Data are reported with a lapse. For example Bloomberg data are indexed by the last date included into the report, which preceeds the releasing date. Make sure to be aligned!
2. Backfilling occurs. That is filling missing data a posteriori.
3. Reinstating values. That is changing a value that wa originally mistaken. For example we have 3 valaues for GDP, the original release and two following corrections. Using the correection in an analysis were these corrections were not yet made creates errors!

### Market Data

Trading activities taking place in an exchange. It is high frequency, not trivial to process and abundant (over 10 TB daily)

### Analytics

Derivative data from fundamental, market or primary analytic data. So the main thing to take in consideration is that these data are pre-processed by somebody. Hence, these might be expensive and also biased!

### Alternative Data

Data produced by individuals (social media, web searches...) or by businesses (transactions,...) or sensors (CCTV, satellites,...)

## Bars

To apply ML we need to parse data in a regular format. We need to create tables with rows or bars. 

The goal is to transform a series of observetions at an irregular frequency into an homogeneous sample

### Time Bars

Obtained by sampling information at fixed time intervals (timestamp, open, close, volume, ...). Even if the most popular they should be avoided.
1. Markets do not process information at a constant time interval. For example open and close hours are more active. Hnece we risk oversampling infomration when the markets are more active.
2. Exhibit poor statistical properties (heteroscedasticity, non-normality,...)

### Tick Bars

Tick bars aggregate data after a fixed number of transactions (like 1,000 trades). This ties the analysis directly to market activity instead of time, making it a proxy for the arrival of new information in the market. Tick bars are more likely to be gaussian distributed, which is a common assumption in many ML methods.

### Volume Bars

Tick bars can have a problem, an order of 10 shares is one tick, while 10 orders of 1 share are 10 transactions. Volume bars solve this issue by sampling every time a pre-defined number of shares. Example, sample prices everytime a contract exchanges 100 units, regardless the number of ticks involved.

### Dollar Bars

Sampling an obs everytime a pre-defined market value is exchanged. Why?
1. If a stock appreciates 100% in X years, it will take half of stockts traded to get the same volume. Hnece, with significant price fluctuations dollar bars appear to be the best choice.
2. The number of outstanding shares changes due to corporate decisions. Dollar bars are robust agains security issues, splits and buybacks


For info on *information-driven bars* see pages 29-32

## Multi-Product Series

If we need to model more instruments with dynamically adjustable weights or deal with products paying irregular coupons/dividends we may have issues. How can we solve it? ... the ETF trick.

### ETF Trick

The goal is to transform any complex multi-product dataset into a single dataset that resembles a total-return ETF. 

Why is this useful? 

Because the code e can always assume that you trade
cashlike products (non-expiring cash instruments), regardless of the complexity and composition of the underlying series.

Suppose we want to develop a strategy to trade futures spread. Some issues arise.

1. Spreads have a vector of weights that changes overtime. Hence, the value of the spread might change even if prices are constant, so you can see you PnL changing because of changing weights (not prices!). You can be mislead that the stragegy is profitable when indeed only weights changed.
2. Spreads are differences or weighted combinations of prices, thus, they can assume negative values. Sometimes handling negative values can be tricky.
3. Different futures contract may have different trading hours. Thus, the spread might not be exactly tradable and you can face latency risk or slippage cause you can't trade all the legs simultaneously.
4. You need to take into account the cost of entering exiting multiple contracts along the duration of the strategy.

How do we solve it? 

Produce a time series that reflects the value of 1 USD invested in a spread. Changes in teh sereis are reflected in the PnL, the series is always postive and the other shortfall are solved. This TS is used to model, generate signal and  trade as if it was an ETF


Here is the notation:

| Symbol                    | Meaning                                         |
| ------------------------- | ----------------------------------------------- |
| $i = 1, ..., I$           | Instrument index (each contract)                |
| $t = 1, ..., T$           | Time index (bar index)                          |
| $B \subseteq \{1,...,T\}$ | Times where rebalancing happens                 |
| $\omega_{i,t}$            | Allocation weight of instrument $i$ at time $t$ |
| $o_{i,t}$, $p_{i,t}$      | Open and close prices                           |
| $\phi_{i,t}$              | USD value of one point (incl. FX)               |
| $d_{i,t}$                 | Carry, dividend, coupon, or cost                |
| $h_{i,t}$                 | Holdings of instrument $i$ at time $t$          |
| $K_t$                     | Total value of portfolio                        |
| $\delta_{i,t}$            | Price change used for PnL                       |
.

where $i = 1, ...,I$ is the instrument index and $t = 1, ...,T$ is the bar index. 

Even if some instruments are not tradable in $[t-1,t]$ they are at least tradable at $t-1$ and $t$. Take a basket of futures with allocation vector $\omega_t$ rebalanced (or rolled) on bars B in {1,..., T}.

We want to simulate how the value of $1 invested in the basket evolves over time:


1. **Holdings** $h_{i t}$: how many of each instrument we hold

$$
h_{i t} = 
\begin{cases}
\frac{\omega_{i t} K_t}{o_{i,t+1} \phi_{i t}} \Bigg/ \sum_{i=1}^I \left| \omega_{i t} \right| & \text{if } t \in B (rebalance) \\
h_{i,t-1} & \text{otherwise}
\end{cases}
$$

- when we rebalnce we recalcualte our holdings
- otherwise we keep the same holdings

2. **Price Change** $\delta_{i t}$: the price movement for each instrument 

$$
\delta_{i,t} = 
\begin{cases}
p_{i t} - o_{i t} & \text{if } (t-1) \in B \\
\Delta p_{i,t} & \text{otherwise}
\end{cases}
$$

3. **Portofolio Update** $K_{i t}$: the PnL is calculated as

$$
K_t = K_{t-1} + \sum_{i=1}^I h_{i t-1} \phi_{i t} \left( \delta_{i t} + d_{i t} \right)
$$

With $K_0 = 1$ the initial AUM. Note that dividends $d_{i t}$ are already embedded in $K_t$. 

4. **Trading Costs and Execution** ....

- **Rebalancing cost**:  

$$c_t = \sum_{i=1}^{I} \left( |h_{i,t-1}| \cdot p_{i,t} + |h_{i,t}| \cdot o_{i,t+1} \right) \cdot \phi_{i,t} \cdot \tau_i$$  

Track this cost separately. It is not embedded in $K_t$, since embedding would cause false PnL (e.g., from shorting). You can treat $c_T$ as a negative dividend** in your code.

- **Bid-ask cost**:  

$$\tilde{c}_t = \sum_{i=1}^{I} |h_{i,t-1}| \cdot p_{i,t} \cdot \phi_{i,t} \cdot \tau_i$$  

This represents the cost of crossing the spread when buying or selling one unit of the synthetic ETF.

- **Volume constraint**:  

$$v_t = \min_i \left\lbrace \frac{v_{i,t}}{|h_{i,t-1}|} \right\rbrace$$  

The least liquid instrument limits how many synthetic spread units can be traded.

### PCA Weights

Another interesting question is: how do we calculate the vector of weights $\omega_t$?

The following method is useful to compute the portfolio allocation vector $\omega_t$ that delivers a user-defined risk distribution across the principal components of the covariance matrix $V$ (covaraince matrix of returns containing volatility and correlation)


Consider an IID Multivariate Gaussian process (like stock returns, changes in bond yields or in options volatilities) for a portfolio of N instruments) having a vector of means $\mu (Nx1)$ and a covariance matrix $V (NxN)$.

How do we proceed?

1. **Spectral decomposition**  

Decompose the covariance matrix $V$:

$$VW = W\Lambda$$

- $W$ contains eigenvectors (principal components giving us the directions of independent risk) 
- $\Lambda$ is a diagonal matrix of eigenvalues (magnitude of risk in a given direction), ordered in descending size

This operation allows us to re-express the portfolio as a combination of orthogonal (uncorrelated) risk sources.

2. **Risk of portfolio allocation $\omega$**  
   
The portfolio (total) risk is:

$$
\sigma^2 = \omega^\top V \omega = \omega^\top W \Lambda W^\top \omega = \beta^\top \Lambda \beta
$$

where:

$$
\beta = W^\top \omega
$$

represents the projection of $\omega$ in the orthogonal (principal component) basis. In other words, the vector of exposure to each component. 

3. **Risk contribution by component**  

Because $\Lambda$ is diagonal:

$$
\sigma^2 = \sum_{n=1}^N \beta_n^2 \Lambda_{n,n}
$$

Define the **risk contribution** of component $n$ as:

$$
R_n = \frac{\beta_n^2 \Lambda_{n,n}}{\sigma^2}
$$

- $\beta_n$ = exposure to component $n$

- $\Lambda_{n,n}$ = variance (risk size) of that component

- $ \beta_n^2 \Lambda_{n,n} $ = how much raw risk from component $n$

$$
\sum_{n=1}^N R_n = 1
$$

So $\{R_n\}_{n=1}^N$ is a **risk distribution** over the principal components. It tells me what % of my total portfolio risk comes from compenent $n$!

4. **Construct $\beta$ from a desired risk distribution $R$**  

Now if it is desired to allocate risk evenly or in a different wa, such as:

- Equal risk: $R_1 = R_2 = ... = R_N = \frac{1}{N}$
- Or maybe: 80% to PC1, 20% to PC2, and 0% to the rest

For this desired target risk distribution $R$, the required projection is:

$$
\beta_n = \sigma \sqrt{ \frac{R_n}{\Lambda_{n,n}} }
$$

5. **Convert $\beta$ back to allocation vector $\omega$**  

$$
\omega = W \beta
$$

Any rescaling of $\omega$ rescales $\sigma$ proportionally, but the **risk distribution $R$ remains unchanged**.

This technique is powerful because it lets you directly design how your portfolio's risk is distributed across statistically independent components (principal components) of market movement.

To summarize, PCA-based portfolio weights are exposures to assets designed to achieve a specific risk distribution across the principal components (uncorrelated risk factors) of the asset returns.

- Weights represent not simple capital allocations but the amount of exposure needed to target risk in each principal component.

- They do not sum to 1 because they are focused on risk allocation, not budget allocation.

- These weights can be positive or negative and may require normalization or scaling before being used in a real portfolio.

- The key goal is to control how risk is distributed across independent sources of variation, rather than how capital is simply split.


### Single Future Roll

When trading futures, rolling from one contract to the next introduces price discontinuities called roll gaps. These gaps can distort the price series if not handled properly. 

**ETF Trick for Spreads**

- The ETF trick treats a spread of futures contracts as a synthetic asset. 
- This works well for multi-legged spreads and can also be applied to a single futures contract (as a 1-legged spread).
- The trick creates a continuous price series by reinvesting PnL and adjusting for roll costs, keeping the series strictly positive.

Alternatively, for one futures only...

**Roll Gaps for Single Futures Contracts**

- Compute the cumulative roll gaps.
- Subtract these cumulative gaps from the raw price series to create a continuous price series without jumps.
- Simpler and more direct for single futures.

Rolled prices are used for simulating PnL. However, raw prices should still be used to size positions and determine capital consumption. Keep in mind, rolled prices can indeed become negative, particularly in futures contracts that sold off while in contango.

In general, we wish to work with non-negative rolled series, in which case we can derive the price series of a $1 investment as follows: (1) Compute a time series of rolled futures prices, (2) compute the return (r) as rolled price change divided by the previous raw price, and (3) form a price series using those returns (i.e.,(1+r).cumprod()).

## Sampling Features

Once a structured dataset is made how can we apply ML?

Well, we need to sample bars to produce features with relevant training examples.

### Sampling for Reduction

Sometime downsampling the features space is useful to fit a ML algo.
Usually we do it by random sampling, tho we could miss out the most important infomration relevant to maximize the predictive power.

### Event-Based Sampling - CUSUM Filter

Sometimes we want to spot how events affect the development of an instrument. One of the most used methods is teh CUSUM filter.

Consider IID observations $y_t$ with $t = 1,..., T$ we define the cumulative sum as:

$$
S_t = \max\left( 0,\ S_{t-1} + y_t - \mathbb{E}_{t-1}[y_t] \right)
$$

With $S_0 = 0$. A signal is generated when $S_t > h$, for some threshold value $h$.  
The filter is used to identify a sequence of positive divergences from any reset level zero.

Clearly, the concept can be extended in a symmetric case:

$$
S_{t+} = \max\left( 0,\ S_{t-1+} + y_t - \mathbb{E}_{t-1}[y_t] \right)
$$

$$
S_{t-} = \max\left( 0,\ S_{t-1-} - y_t + \mathbb{E}_{t-1}[y_t] \right)
$$

$$
S_t = \max\left( S_{t+},\ S_{t-} \right)
$$

