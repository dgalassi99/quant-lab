# MODELLING - HYPERPARAMETERS TUNING WITH CV

HP tuning is important in ML to avoid the risk of overfitting. We have seen that with financial observation a simple k-fold CV fails, and a Purged k-fold (purging and embargoing) is a potential solution. We now investigate how to use this technique for HP tuning.

## Grid Search CV

Grid search CV is an extensive reasearch of all the possible combinations of user-defined HPs aimed at maximizing CV performances. This method is already implemented in Scikit which accept a CV generator as argument. But we need to use our PurgedKFold class to prevent overfitting and data leakage.

We use the clfHyperFit function:
- **fit_params: used to pass sample_weight and param_grid
- use as scoring F1 for metalabelling applications as high accuracy or neg_log_loss is high when we have an imbalance with a lot of 0 classes
- use indeed accuracy or neg_log_loss in the rother cases as we are equally interested in predicting all cases

A limitation of Scikit is that we can't use sample_weight when using Pipeline class. We can overwrite by using our own class MyPipeline which inherits and expands the Pipeline Scikit class. 

## Randomized Search

Wehn the number of combinations of parameters >> 1 a Grid Search CV experiences a computational explosion. So we sample combinations of parameters randomly (follwing a distribution of possible combinations) as we can:
- control for the total number of combinations
- having irrelevant parameters will not increase the computational burden 

We expand the clfHyperFit function.

But how do we sample these combinations of parameters?

### Log-Uniform Distributions

A lot of ML algos accept only non-negative params. Take for example C in SVC or gamma in RBF kernel. We could draw the values of these parameters from a uniform distirbution, say bounded between 0 and 100. This implies that 99% of the params are > 1. Ok, but this is not (necessarily) the best way when the model has a non linear response to parameters changes.

Why? In SVC an increase of C from 0.01 to 1 or from 1 to 100 can have different responsiveness. If we draw from U[0,100] is quite inefficient. Hence, we prefer drawing from the distribution og the log of those values. That is a "log-uniform distribution".

An rv x follows a log-unif. distrib. between $a>0$ and $b>a$ iff $log[x] ~ U[log(a),log(b)]$. This has a ...

CDF:
- $F[x] = \frac{log[x]-log[a]}{log[b]-log[a]}$ for $a <= x <= b$
- $F[x] = 0$ for $x < a$
- $F[x] = 1$ for $x > b$

PDF: 
- $f[x] = \frac{1}{xlog[b/a]}$ for $a <= x <= b$
- $f[x] = 0$ for $x < a$
- $f[x] = 0$ for $x > b$

## Scoring and HPs Tuning

As we said, we use F1 for metalabelling and accuracy/neg_log_loss for other applications. Tho, using neg_log_loss is better in the second case. Here is why.

Suppose your ML strategy tells you to buy a security with high confidence. Hence, yuo go long with a big size. If this prediction was wrong, you lose money. Now, accuracy simply count right vs incorrect predictions and treats mistakes equally (regardless the model confidency).

If the model says:
- buy with 99% confidence and it was wrong --> accuracy counts +1 wrong
- buy with 50.001% confidence and it was wrong --> accuracy counts +1 wrong

Investment strategies profit from positioning in high confidence cases. Gains in low probability moves do not offset losses in high confidence picks (especially if you size the bets based on this confidence!). Hence, accuracy is not a great metric!

Differently, log-loss (entropy-loss) computes the log-likelihood of the classifier given the true label which takes prediction probs. into account.

$$ L[Y,P] = - log(Prob[Y|P]) = -N^-1  finsh formula $$
