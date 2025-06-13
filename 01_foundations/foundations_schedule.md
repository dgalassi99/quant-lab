# 01-Foundations

This section covers the basics of Python, math, and statistics for quant trading.

## 1

### Theory
- [x] organize the structure of the git repository for month 1-3
- [x] study *Quantitative Trading* - Ch 1
- [x] read/repeat *Python for Data Analysis* - Ch 5
- [x] study *Options, Futures, and Other Derivatives* - Ch 1
- [x] What is quant trading? Why using Python? What is the workflow (retail vs institutional)?
- [x] quantopian lect. 1
### Practice
- [x] write a script to download and clean historical OHLCV data for AAPL, SPY, and BTC using yfinance
- [x] create a function that computes basic stats: return, volatility, rolling average
- [x] plot 30-day rolling mean & volatility — what patterns do you observe?

## 2

 ### Theory
- [x] study *Quantitative Trading* - Ch 2
- [x] read/repeat *Python for Data Analysis* - Ch 6
- [x] read documentation about *rolling, expanding, resample, shift*
- [x] build moving averages (MA, EMA) and a signal generator for long/short based on MAs crossing 
- [x] What technical indicators are? What are pros/cons of simple rule-based strategies
- [x] quantopian lect. 2-3
### Practice
- [x] implement SMA/EMA with flexible window sizes
- [x] create a crossover strategy with signals (long = 1, short = -1)
- [x] test various combinations (20/50, 50/200) and plot the signals over price
- [x] compare strategy returns vs. buy-and-hold — what do you notice?
 

## 3

### Theory
- [x] study *Quantitative Trading* - Ch 3
- [x] study *Trading and Exchanges* - Ch 1,2
- [x] implement a strategy logic by: (1)create position column: 1 if in trade, 0 otherwise; (2) simulate basic entry/exit logic; (3) create PnL, cumulative returns, and basic performance metrics 
- [x] more: Market Structure (exhanges, order books), Order types (market, limit, stop iceberg)... you can watch https://www.youtube.com/watch?v=DDtWLBAgG1s
- [x] quantopian lect. 4
### Practice
- [x] add a returns column and calculate strategy performance
- [x] plot rolling Sharpe and compare it to passive exposure
- [x] simulate order types with slippage – how does PnL change?

## 4

### Theory
- [x] read/repeat *Python for Data Analysis* - Ch 7
- [x] backtest the MA strategy by: (1)Track daily returns with pct_change() * position.shift(1); (2) Plot equity curve; (3) Add simple performance stats (Sharpe, win rate, max drawdown); (4) Try on more equities ;(5) Prepare a nice ipynb
- [x] more: What could go wrong in real execution? (slippage, latency, false signals); brainstorm ways to improve the strategy
### Practice
- [x] write a helper function to compute key metrics (Sharpe, Max DD, CAGR, Win Rate)
- [x] try the strategy on at least 3 more tickers — do results generalize?
- [ ] test the effect of adding volatility filters or volume conditions --> when i add these filters the trading strategiees gives stranfge signals --> i have to check once again

## Final Deliverables
- A notebook titled *Moving_Average_Strategy.ipynb*
- A 1-page "strategy brief" with: (1) Description; (2) Entry/exit logic; (3) Performance stats; (4) Charts & commentary
- A document containing most important parts of the chapters red and if necessary of the extra questions/brainstorm activities
