# MODELLING - FEATURE IMPORTANCE

## The Importance of Feature Importance

The most common mistake one can make is to take data, feed it to an ML algorithm, backtest the predictions and repeat until a nice-looking backtest comes out. 
As of right now, the omly important thing is to consider that this is an analysis to be performed before any backtest and that *backtesting is not a reseach toll, feature importance is*.

## Feature Importance with Substitution Effects 

Substitution effect takes place when the importance of a given feature is reduced by the presence of other related features. This is very similar to the issue of multicollinearity
and can be solved by applying PCA on the raw features and perfrom the feature importnace on the ortogonal predictors.

### Mean Decrease Impurity (MDI)

It is an in-sample method for tree-based models. At each node of decision trees, the selected feature splits the subset to decrease the node impurity. Thus, for each tree,
we can get how much of the overall impurity decreased can be assigned to each predictor. Finally, average out this result over the entire forest to rank out predictors.

Some considerations:

- Some features might be systematically ignored so we can set max_features = 1 when using RF so that every feature is given a chance (randomly selected)
- The procedure is in-sample, nothing we can say above the predictive power
- MDI is normalized such that the sum of importances is 1
- MDI is the default FI score in sklearn library

### Mean Decrease Accuracy (MDA)

MAD is a slow but out-of-sample method. It fits a classifier, derives its OOS performances, and permutates the column of the feature matrix one by one
deriving the performances after each column permutation. The importance of a feature is derived as the loss in performance due to its column permutation.

Some considerations:

- MDA is not limited to accuracy as the sole performance score and the importance mught be negative (the feature is detrimental for the model forecasting power) 
- MDA might conclude that all features ar non-important (differently from MDI) because it is an OOS metric
- The CV must be purged and embargoed

## Feature Importance without Substitution Effects 

Substitution effects migt lead us to remove importnat features just because of redundancy. This is not a problem in the context of prediction, but it is if we are trying to 
understand/improve a model based on FI.

### Single Feature Importance

SFI is an OOS method computing the OOS performance score of each feature in isolation. Clearly joints effect and hierarchical importances are lost in SFI.

### Orthogonal Features

Substitution effects cam dilute the FIs measured with MDI or MDA. A partial solution is to orthogonalize the feature space a priori using PCA.

Take a matrix $X_{t,n}$ with $t=1,...,T$ observations and $n=1,...,N$ variables. 

- Standardize the feature matrix: $Z_{t,n} = \frac{X_{t,n}-\mu_n}{\sigma_n}$
- Compute eigenvalues and eigenvectors $Z^TZW = W\Lambda$ with $\Lambda$ diagonal matrix with sorted eigenvalues ($\lambda_1>=,...>=\lambda_N$) and $W$ is a $NxN$
matrix of eigenvectors (orthonormal $W^TW=I$) 
- Derive the principal components $P=ZW$ where the columns of $P$ are orthogonal $P^TP=(ZW)^T(ZW)=Z^TW^TZW=W^TW \Lambda W^TW=I \Lambda I = \Lambda$

Besides addressing subs. effects orthogonalization: (1) can be used for dimensionality reduction; (2) we can better understand the structure of the data. PCA is an
unsupervised learning, hence, it ranks features eigenvalues regardless of any possible overfitting (differently from MDI/MDA). Hence, if MDI/MDA importances agree with PCA this 
is a confirmation signal --> you can check eigenval/FI correlation or Kendall's tau (the closer to 1 the better).

