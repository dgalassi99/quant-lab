# ADVANCES IN FINANCIAL MACHINE LEARNING: RESUME

## SECTION 1 - DATA ANALYSIS

### Financial Data Structures

#### Types of Financial Data

**Fundamental Data**

Mostly accounting data reported quarterly. Those have low frequency and can be used in coupling with higher frequency data.

 Typical issues?

1. Data are reported with a lapse. For example Bloomberg data are indexed by the last date included into the report, which preceeds the releasing date. Make sure to be aligned!
2. Backfilling occurs. That is filling missing data a posteriori.
3. Reinstating values. That is changing a value that wa originally mistaken. For example we have 3 valaues for GDP, the original release and two following corrections. Using the correection in an analysis were these corrections were not yet made creates errors!

**Market Data**

Trading activities taking place in an exchange. It is high frequency, not trivial to process and abundant (over 10 TB daily)

**Analytics**

Derivative data from fundamental, market or primary analytic data. So the main thing to take in consideration is that these data are pre-processed by somebody. Hence, these might be expensive and also biased!

**Alternative Data**

Data produced by individuals (social media, web searches...) or by businesses (transactions,...) or sensors (CCTV, satellites,...)

#### Bars

To apply ML we need to parse data in a regular format. We need to create tables with rows or bars. 

The goal is to transform a series of observetions at an irregular frequency into an homogeneous sample

**Time Bars**

Obtained by sampling information at fixed time intervals (timestamp, open, close, volume, ...). Even if the most popular they should be avoided.
1. Markets do not process information at a constant time interval. For example open and close hours are more active. Hnece we risk oversampling infomration when the markets are more active.
2. Exhibit poor statistical properties (heteroscedasticity, non-normality,...)

**Tick Bars**

Tick bars aggregate data after a fixed number of transactions (like 1,000 trades). This ties the analysis directly to market activity instead of time, making it a proxy for the arrival of new information in the market. Tick bars are more likely to be gaussian distributed, which is a common assumption in many ML methods.

**Volume Bars**

Tick bars can have a problem, an order of 10 shares is one tick, while 10 orders of 1 share are 10 transactions. Volume bars solve this issue by sampling every time a pre-defined number of shares. Example, sample prices everytime a contract exchanges 100 units, regardless the number of ticks involved.

**Dollar Bars**

Sampling an obs everytime a pre-defined market value is exchanged. Why?
1. If a stock appreciates 100% in X years, it will take half of stockts traded to get the same volume. Hnece, with significant price fluctuations dollar bars appear to be the best choice.
2. The number of outstanding shares changes due to corporate decisions. Dollar bars are robust agains security issues, splits and buybacks


For info on *information-driven bars* see pages 29-32

#### Multi-Product Series

If we need to model more instruments with dynamically adjustable weights or deal with products paying irregular coupons/dividends we may have issues. How can we solve it? ... the ETF trick.

##### ETF Trick

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


Suppose we have a series of bars with the following columns:

- $o_{i t}$ is the open price
- $p_{i t}$ is the close price
- $f_{i t}$ is the USD value of one point of instrument (including foreign excahnge rate) 
- $v_{i t}$ is the volume
- $d_{i t}$ is the carry, dividend, coupon paid, ... or other costs.

where $i = 1, ...,I$ is the instrument index and $t = 1, ...,T$ is the bar index. 

Even if some instruments are not tradable in $[t-1,t]$ they are at least tradable at $t-1$ and $t$. Take a bastek of futures with allocation vector $\omega_t$ rebalanced (or rolled) on bars B in {1,..., T}, the 1 USD investment value ${K_t}$ is: 

$$
h_{i,t} = 
\begin{cases}
\frac{\omega_{i,t} K_t}{o_{i,t+1} \phi_{i,t}} \Bigg/ \sum_{i=1}^I \left| \omega_{i,t} \right| & \text{if } t \in B \\
h_{i,t-1} & \text{otherwise}
\end{cases}
$$

$$
\delta_{i,t} = 
\begin{cases}
p_{i,t} - o_{i,t} & \text{if } (t-1) \in B \\
\Delta p_{i,t} & \text{otherwise}
\end{cases}
$$

$$
K_t = K_{t-1} + \sum_{i=1}^I h_{i,t-1} \phi_{i,t} \left( \delta_{i,t} + d_{i,t} \right)
$$
