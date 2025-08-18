# 03-Machine-Learning-in-Finance-I

This section introduces supervised ML techniques applied to financial time series, signal labeling, and model evaluation.

---

## 9  

### 📚 Theory
- [x] Study *Advances in Financial ML* – Ch.2,3
- [x] Study *The Science of Algorithmic Trading* – Ch.2  
- [x] Review Quantopian Lectures 11–12  
- [x] Understand why traditional ML struggles in finance  
- [x] Learn about financial features and data stationarity

### 💻 Practice
- [ ] Download historical data and compute financial features (returns, vol, momentum, etc.)  
- [ ] Normalize/standardize features, handle NaNs, and resample for ML  
- [ ] Use `train_test_split()` to create your dataset  

---

## 10  

### 📚 Theory
- [ ] Study *Advances in Financial ML* – Ch.4  
- [ ] Review Quantopian Lectures 13–14  
- [ ] Learn about labeling: fixed horizon, triple-barrier, and event-driven  
- [ ] Study common pitfalls: lookahead bias, imbalance, data leakage

### 💻 Practice
- [ ] Implement binary labels using return thresholds  
- [ ] (Optional) Explore triple-barrier labeling as per López de Prado  
- [ ] Visualize label distributions and class imbalance  

---

## 11  

### 📚 Theory
- [ ] Study *Advances in Financial ML* – Ch.5  
- [ ] Study *The Science of Algorithmic Trading* – Ch.7  
- [ ] Quantopian Lectures 15–16  
- [ ] Learn about Random Forests and Ensemble Methods (Bagging, Boosting)  
- [ ] What is feature importance? Why use SHAP in finance?

### 💻 Practice
- [ ] Train a Random Forest Classifier on labeled data  
- [ ] Use SHAP to evaluate feature importance  
- [ ] Visualize confusion matrix, precision, recall, and F1 score  

---

## 12  

### 📚 Theory  
- [ ] Read additional materials from López de Prado on model overfitting and cross-validation  
- [ ] Explore time series cross-validation and walk-forward CV  
- [ ] Discuss interpretability and trust in ML models

### 💻 Practice  
- [ ] Implement XGBoost with cross-validation  
- [ ] Compare model performance: Random Forest vs XGBoost  
- [ ] Tune hyperparameters with `GridSearchCV` or `optuna`  
- [ ] Build a dashboard: accuracy, ROC-AUC, feature plots

---

## 📦 Final Deliverables  
- ✅ Notebook: `ML_Trading_Classifier.ipynb`  
- ✅ Labeling script for fixed threshold & triple-barrier  
- ✅ Evaluation report:
  1. Features used  
  2. Labeling method  
  3. Classifier results (metrics + plots)  
  4. SHAP insights  
- ✅ Summary doc:
  - Key insights from *Advances in Financial ML* Ch.2, 4, 5  
  - *Science of Algo Trading* Ch.2 & 7  
  - Notes on Quantopian Lectures 11–16  

