# MODELLING - ENSEMBLE METHODS

## Sources of Errors

ML models suffers three type of errors:

- **Bias:** the algorithm fails to recognize important patters and the model is said to be underfit.
- **Variance:** the algorith has overfit the training set, hence it is very sensible to changes in training data. The model has miskatenly learned noise instead of important patterns.
- **Noise:** irreducible error why can't be explained by any model.

Given a training set {$x$} and outcome {$y$} suppose there exist a function such that $y = f(x) + \epsilon$. This $\epsilon$ models the noise such that $E(\epsilon_i) = 0$ and $E(\epsilon_i^2) = \sigma_{\epsilon}^2$. The goal is to estimate the function $g(x)$ that best fits $f(x)$ by minimizing the variance of the estimator $E((y_i-g(x_i))^2)$. We can decompose this term as:

$$ E((y_i-g(x_i))^2) = E((g(x_i))-f(x_i)))^2 + V(g(x_i)) + \sigma_{\epsilon}^2 = BIAS^2 + VARIANCE + NOISE $$

Ensemble methods combine different algorithms that perform better than individual ones --> improving the bias-var trade-off.

## Bootstrap Aggregation (Bagging)

Bagging is an effective way to reduce variance. First, we random sample with replacement N subtraining sets from the main training set and then fit N estimators (idependently --> in parallel fitting) on these N subsets. Then, the ensemble forecast is simply the average of the N-individual forecasts. In the case of categorical variables, the probability that an observation belongs to a class is given by the proportion of estimators that classify this observation as belonging to that class (majority voting)

### Variance Reduction

The variance of a bagging estimator is:

$$ \bar{\sigma}\bar{\rho} + \bar{\sigma}(1-\bar{\rho})/N  $$

Where $\bar{\sigma}$ is average variance of a single estimator, and $\bar{\rho}$ the correlation between estimators. Thus, we should note that.

- The variance of a single estimator (on a subset) is higher than the one of the estimator on the full training (what we get without bagging)  set as the num. of obs. is smaller. The advantage comes from avegaring out on a big number of subestimators.
- When correlation  $\bar{\rho}$ is close to 1 it is like using only one estimator many times, hence, we lose the bagging advantage. Oppositely, variance reduction is maximimized (linearly decreases with N) when estimators are completely non correlated ($\bar{\rho} = 0$).

This concept of estimators correlation is fundamental in bagging techniques. We saw, for example, how sequantial bootstrapping aims at producing samples as independent as possible --> reducing  $\bar{\rho}$ --> reducing total model variance.

### Improved Accuracy

Take a classfier that makes prediction on $k$ classes by majority voting among N classfiers where predictions can be labelled as {1,0} if correct or wrong. 

- The accuracy of a single classifier is the probability $p$ ($p>1/k$ which is random guessing) of labelling a prediction as 1. On average we get $Np$ predictions labelled as 1 with varaince $Np(1-p)$
- By the LLN the amjority voting will reflect the true signal rather than the noise of a single classifier, thus, as N grows, the probability that the majority is correct gets closer to 1 (provided that $p>1/k$) --> bagging improves accuracy as it cancels out the mistake of single classifiers!

Note: it usually easierto achieve a decrease in $\bar{\rho}$ than getting $p<1/k$ --> bagging is much better at reducing variance than improving accuracy --> start with base classfier with low bias and high varaince and use bagging to reduce the last.

### Observation Redundancy

We saw that financial observations are far from being IID. This have two detrimental effects on bagging:

1. Samples drawn with replacement are more likely to be identical (high correlation) --> bagging is not able to reduce the variance even if we increase N
2. OOB accuracy is inflated because random sampling with replacement places in the training sets observations whihc are very similar to the out-of-bag ones.

In fact a proper k-fold cross validation without shuffling before partitioning will show a test accuracy much lower than th ebagging oob. It is advisable to set StratifiedKFold(n_spiltsk,shuffle=False) when using this class, cross-validate teh bagging classifier and ignore oob accuracy results. A low $k$ is preferred reducing the chance to reduc the likelihood of pracing in the test set observation very similar to the ones in the training set.

## Random Forest

RF follows baggin structure but introduces another layer of randomness by reducing the number of available features at the moment of node splits. This further decorrelates the estimators as it will be less likey to have similar splits in all subtrees. Another important traits of RF is its ability to output feature importance and oob accuracy estimates. In practice with non-IID obesrvations we can implement some of th efollowing tricks:

- Set max_features to a low value by forcing the trees to be as different as possible
- Set the regularization parameter min_weight_fraction_leaf to a large value (5%) such that oob accuracy converges to out-of-sample (k-fold) accuracy
- Use BaggingClassifier on DecisionTreeClassifier where max_samples is set tot the average uniqueness (avgU) between samples:
  - clf=DecisionTreeClassifier(criterion='entropy',max_features='auto',class_weight='balanced')
  - bc=BaggingClassifier(base_estimator=clf,n_estimators=1000,max_samples=avgU,max_features=1.)
Use BaggingClassifier on RandomForestClassifier where max_samples is set tot the average uniqueness (avgU) between samples:
  - clf=clf=RandomForestClassifier(n_estimators=1,criterion='entropy',bootstrap=False,class_weight='balanced_subsample')
  - bc=BaggingClassifierbc=BaggingClassifier(base_estimator=clf,n_estimators=1000,max_samples=avgU,max_features=1.)
- Modify the RF class to replace standard bootstrapping with sequential bootstrappin

## Boosting

Boosting aims at reducing the bias of low-variance models. It generates one ranodm subsample and fits one estimator. If this estimator perform better thana threshold (50% in a binary classfier - better than random) the estimator is kept, otherwise discarded. Give more weight to the misclassified observations and repeat the pevious step until N estimators are produced. The ensemble forecats is the weighted average forecats over the N individial forecats, where weights are functions of indicial estimators accuracies. 

## Bagging vs Boosting in Finance

*What are the main differences?* Well, boosting...

- Fits indivisual classifiers sequentially
- Poor-performing classifiers are dismissed
- Observations are weighted differently at each iteration
- The ensemble forecast is a weighted average of individual learners
- Addresses underfitting while bagging overfitting --> we generally have to deal with overfitting (bagging is usually preferred)
- Can't be parallelized as the model is built sequentially
