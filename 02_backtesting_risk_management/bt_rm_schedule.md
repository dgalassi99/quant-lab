# 02-Backtesting-and-Risk

This section introduces structured backtesting, risk management, and robust strategy evaluation.

---

## 5

### 📚 Theory
- [x] Study *Quantitative Trading* – Ch.4,5   
- [x] Watch Quantopian Lectures 5–6  
- [x] What makes a good backtest? Understand lookahead bias, overfitting, and market regime change.  
- [x] Learn the concept of position sizing and risk-adjusted returns.

### 💻 Practice
- [x] Install and explore `bt` or `backtrader` — try running a basic MA strategy in either  
- [x] Write a modular backtesting class with: signal input, execution logic, performance stats  
- [x] Create position-sizing logic based on volatility or fixed allocation  

---

## 6 

### 📚 Theory
- [x] Study *Quantitative Trading* – Ch.6    
- [x] Review stop-loss and drawdown management  
- [x] Watch Quantopian Lectures 7–8  
- [x] Discuss the difference between simulation and real execution (slippage, fill models, etc.)

### 💻 Practice
- [x] Implement stop-loss and take-profit logic in your backtest  
- [x] Add trailing stop and maximum drawdown exit conditions  
- [x] Simulate slippage and transaction costs, observe effect on performance  

---

## Extra: Focus on *ML for Asset Managers*
I find that this book could be useful for general knowledge and mostly also for the next section of the studies which is based on ML in finance.

Hence, I will go through some or all the chapters of the book and create an ipynb file where I follow what is done and try to sole the exercises proposed --> Here is the link: https://github.com/dgalassi99/quant-trading-self-study/blob/main/00_books/Machine_Learning_for_Asset_Managers.ipynb

---
## 7

### 📚 Theory
- [ ] Study *ML for Asset Managers* – Ch.3 (focus: evaluation and validation)  
- [ ] Read *Options, Futures, and Other Derivatives* – Ch.2  
- [ ] Watch Quantopian Lecture 9  
- [ ] What is walk-forward validation? Why is it crucial in live trading?  
- [ ] How to split datasets in-sample, out-of-sample, and test

### 💻 Practice
- [ ] Apply walk-forward validation to your MA strategy  
- [ ] Write a function that iterates through rolling windows and records performance  
- [ ] Visualize out-of-sample performance stability  

---

## 8

### 📚 Theory  
- [ ] Study *Options, Futures, and Other Derivatives* – Ch.3  
- [ ] Watch Quantopian Lecture 10  
- [ ] Explore how futures/options contracts can be used in strategy hedging  
- [ ] What are leverage, margin, and risk exposure?

### 💻 Practice
- [ ] Incorporate leverage in your strategy simulation  
- [ ] Build a dashboard that shows: cumulative returns, drawdown, volatility, Sharpe, exposure  
- [ ] Compare strategies with and without drawdown protection  

---

## 📦 Final Deliverables  
- ✅ A backtesting engine in Python (can use `bt` or `backtrader`)  
- ✅ A notebook: `Backtest_MA_Strategy_with_Risk_Controls.ipynb`  
- ✅ A 1-pager including:
  1. Strategy description and enhancements (e.g., stop-loss)  
  2. Performance overview with walk-forward validation  
  3. Plots, summary tables, and lessons learned  
- ✅ A short doc summarizing key insights from:
  - *Quantitative Trading* Ch.5–6  
  - *ML for Asset Managers* Ch.3  
  - *Inside the Black Box* Ch.3–4  

