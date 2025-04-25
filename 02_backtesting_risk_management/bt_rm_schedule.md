# 02-Backtesting-and-Risk

This section introduces structured backtesting, risk management, and robust strategy evaluation.

---

## Week 5  
**17/05/2025–24/05/2025**  

### 📚 Theory
- [ ] Study *Quantitative Trading* – Ch.4,5  
- [ ] Read *Inside the Black Box* – Ch.3  
- [ ] Watch Quantopian Lectures 5–6  
- [ ] What makes a good backtest? Understand lookahead bias, overfitting, and market regime change.  
- [ ] Learn the concept of position sizing and risk-adjusted returns.

### 💻 Practice
- [ ] Install and explore `bt` or `backtrader` — try running a basic MA strategy in either  
- [ ] Write a modular backtesting class with: signal input, execution logic, performance stats  
- [ ] Create position-sizing logic based on volatility or fixed allocation  

---

## Week 6  
**25/05/2025–01/06/2025**  

### 📚 Theory
- [ ] Study *Quantitative Trading* – Ch.6  
- [ ] Read *Inside the Black Box* – Ch.4  
- [ ] Review stop-loss and drawdown management  
- [ ] Watch Quantopian Lectures 7–8  
- [ ] Discuss the difference between simulation and real execution (slippage, fill models, etc.)

### 💻 Practice
- [ ] Implement stop-loss and take-profit logic in your backtest  
- [ ] Add trailing stop and maximum drawdown exit conditions  
- [ ] Simulate slippage and transaction costs, observe effect on performance  

---

## Week 7  
**02/06/2025–09/06/2025**

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

## Week 8  
**10/06/2025–17/06/2025**

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

