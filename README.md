# quant-trading-self-study
My 9-month self-directed program to deep dive into financial engineering, quant analysis and trading strategies

## Useful Links
- GitHub repository with 56 lectures (video+notebook): https://gist.github.com/ih2502mk/50d8f7feb614c8676383431b056f4291
- GitHib with links to libraries, strategies and books: https://github.com/paperswithbacktest/awesome-systematic-trading

## Structure of the programme

_Note: The programme could (most likely) dinamically change due to adjustments or just because it will take me more time to complete certain steps or, finally, because I will deepend some fields and be bored by others_

Now, as written in the first lines I am planning to learn and improve my quantitative skills following a 9-months self study plan (10-15 hours/week). The plan will be modified as I collect more info or change ideas and is based on a MSc I would have love to follow (I could not due to work related obligations) which is the MFE from EPFL (Lausanne, CH).

Well... let's start

### Month 1: Foundations of Quantitative Trading & Python
Goals: Understand core principles of quant trading, basic data workflows, and Python for finance.\

Books & Chapters:
- Quantitative Trading - E. Chan: Ch.1-3
- Python for Data Analysis - W. McKinney: Ch.5-7
- Trading and Exchanges - L. Harris: Ch.1-2
- Options, Futures, and Other Derivatives - J. Hull: Ch.1

Tasks:
- Learn pandas, numpy, yfinance
- Reproduce moving average crossover strategy
- Understand market structure & order types

More:
- Lecture 1-4 quantopian link

### Month 2: Backtesting and Risk Management
Goals: Build backtests and understand execution and risk.
Books & Chapters:
- Quantitative Trading - E. Chan: Ch.5-6
- Inside the Black Box - R. Narang: Ch.3-4
- ML for Asset Managers - M. López de Prado: Ch.3
- Options, Futures, and Other Derivatives - J. Hull: Ch.2-3
  
Tasks:
- Use bt or backtrader for backtesting
- Walk-forward validation
- Implement stop-loss, drawdown logic


More:
- Lecture 5-10 quantopian link

### Month 3: Machine Learning in Finance I
Goals: Apply ML methods to trading signals.
Books & Chapters:
- Advances in Financial ML - M. López de Prado: Ch.2, 4, 5
- The Science of Algo Trading - R. Kissell: Ch.2, 7
- GitHub repo: ih2502mk's notebooks

Tasks:
- Feature engineering & importance (SHAP)
- Implement Random Forest, XGBoost
- Labeling return-based outcomes

More:
- Lecture 11-16 quantopian link

### Month 4: Time Series & Financial Econometrics
Goals: Forecast returns & volatility.
Books & Chapters:
- Intro to Time Series Forecasting - J. Brownlee (select chapters)
- Python for Data Analysis - Ch.11
- Quantitative Trading (review time series parts)
- Options, Futures, and Other Derivatives - J. Hull: Ch.11-12

Tasks:
- Use ARIMA/GARCH models
- Perform trend decomposition
- Evaluate MAE/RMSE

More:
- Lecture 17-20 quantopian link

### Month 5: Advanced ML & Portfolio Optimization
Goals: Optimize portfolios and sizing models.
Books & Chapters:
- Advances in Financial ML - Ch.7, 10
- ML for Asset Managers - Ch.5
- The Science of Algo Trading - Ch.11
- The Concepts and Practice of Mathematical Finance - M. Joshi: Ch.1-2

Tasks:
- Implement Kelly criterion, risk parity
- Cross-validated ML pipeline
- Build an ML-optimized portfolio

More:
- Lecture 21-26 quantopian link

### Month 6: Stochastic Calculus and Derivatives Pricing
Goals: Learn core continuous-time finance and option pricing theory.
Books & Chapters:
- Stochastic Calculus for Finance I: The Binomial Asset Pricing Model - S. Shreve: Entire Book
- Stochastic Calculus for Finance II: Continuous-Time Models - S. Shreve: Ch.1-4, 7-8
- Options, Futures, and Other Derivatives - J. Hull: Ch.8-13
- The Concepts and Practice of Mathematical Finance - M. Joshi: Ch.3-7

Tasks:
- Brownian motion & Itô calculus
- Derive and implement Black-Scholes
- Replicate binomial option pricing model

More:
- Lecture 27-32 quantopian link

### Month 7: Numerical Methods for Finance
Goals: Solve pricing and optimization problems numerically.
Books & Chapters:
- Numerical Methods in Finance and Economics (P. Brandimarte: Ch.1-5, 9, 10)
- Options, Futures, and Other Derivatives - J. Hull: Ch.14-17
- The Concepts and Practice of Mathematical Finance - M. Joshi: Ch.8-9

Tasks:
- Implement Monte Carlo simulations for option pricing
- Build finite difference models for PDEs (Black-Scholes)
- Understand matrix-based optimization problems

More:
- Lecture 33-38 quantopian link

### Month 8: Strategy Integration & Project Design
Goals: Develop and evaluate a complete trading strategy.

Tasks:
- Combine ML + time series + risk + pricing ideas
- Run backtests, stress tests, and robustness checks
- Use GitHub to write clean documentation
- Begin report and visualization dashboards

More:
- Lecture 39-44 quantopian link

### Month 9: Capstone Project & Interview Prep
Goals: Finalize your project and prepare for quant interviews.
Tasks:
- Complete a polished capstone with Jupyter + PDF write-up
- Mock interviews: stats, brainteasers, ML, finance
- 
Resources:
- Heard on the Street - T. Crack (Interview)
- A Practical Guide to Quant Finance Interviews - X. Zhou
- QuantStart mock questions
- GitHub repo: ih2502mk's notebooks

More:
- Lecture 45-56 quantopian link
