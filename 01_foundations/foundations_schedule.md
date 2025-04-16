# 01-Foundations

This section covers the basics of Python, math, and statistics for quant trading.

## Week 1
15/04/2025-22/04/2025

- [ ] organize the structure of the git repository
- [ ] study *Quantitative Trading* - Ch 1
- [ ] read/repeat *Python for Data Analysis* - Ch 5
- [ ] fetch historical data from yfinance
- [ ] more: What is quant trading? Why using Python? What is the workflow (retail vs institutional)?

## Week 2
23/04/2025-30/04/2025
 
- [ ] study *Quantitative Trading* - Ch 2
- [ ] read/repeat *Python for Data Analysis* - Ch 6
- [ ] read documentation about *rolling, expanding, resample, shift*
- [ ] build moving averages (MA, EMA) and a signal generator for long/short based on MAs crossing 
- [ ] more: What technical indicators are? What are pros/cons of simple rule-based strategies


## Week 3
01/05/2025-08/05/2025 

- [ ] study *Quantitative Trading* - Ch 3
- [ ] study *Trading and Exchanges* - Ch 1,2
- [ ] implement a strategy logic by: (1)create position column: 1 if in trade, 0 otherwise; (2) simulate basic entry/exit logic; (3) create PnL, cumulative returns, and basic performance metrics 
- [ ] more: Market Structure (exhanges, order books), Order types (market, limit, stop iceberg)... you can watch https://www.youtube.com/watch?v=DDtWLBAgG1s

## Week 4
09/05/2025-16/05/2025

- [ ] read/repeat *Python for Data Analysis* - Ch 7
- [ ] backtest the MA strategy by: (1)Track daily returns with pct_change() * position.shift(1); (2) Plot equity curve; (3) Add simple performance stats (Sharpe, win rate, max drawdown); (4) Try on more equities ;(5) Prepare a nice ipynb
- [ ] more: What could go wrong in real execution? (slippage, latency, false signals); brainstorm ways to improve the strategy

## Final Deliverables
- A notebook titled *Moving_Average_Strategy.ipynb*
- A 1-page "strategy brief" with: (1) Description; (2) Entry/exit logic; (3) Performance stats; (4) Charts & commentary
- A document containing most important parts of the chapters red and if necessary of the extra questions/brainstorm activities
