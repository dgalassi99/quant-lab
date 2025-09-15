# MODELLING - CROSS-VALIDATION IN FINANCE

## The Goals of CV

CV is the process of splitting the starting data set into a training and testing sets such that each observation belongs to one and one set only in order 
to prevent data leakage. One of the most used CV scheme is the k-fold CV where the train/test split is performed as follows:

- Partition the data into $k$ subsets
- For $i =1,..., k$
  - Train on all subsets other than $i$
  - Test on subset $i$
 
## Why k-fold CV Fails in Finance

First, we have seen that observations are not IID. Second, the testing set is used multiple times for model development leading to selection bias. But, let's now focus
on th efirst issue.

Leakage takes place when the training set contains infomration that should only be present in the test set. Take an autocorrelated feature $X$ associated to labels $Y$ 
formed on overlapping data. Now: (1) because of autocorrelation, $X_t ~ X_{t+1}$; (2) because labels come from overlapping data points, $Y_t ~ Y_{t+1}$. If $t$ and $t+1$ 
are placed in two different sets we have data leakage, and the performance of the model is inflated. Moreover, this inflation occurs also if X is a feature with low predictive power 
(low-importance feature) which leads to false discoveries.

*How do we reduce the likelhood of leakage?*

1. Drop from the training set any observation $i$ where $Y_i$ is a function of information used to determine $Y_j$ and $j$ belongs to the test set. For example,
$i$ and $j$ should not span overlapping periods.
2. Avoid overfitting the classifier by using early stopping for base estimators, bagging while contorlling for oversampling on redundant observations  (max_sample = avgU
and use sequential boostrap).

## Purged k-fold CV

The process of removing from the training set the observations whose labels overlap in time with the ones in the test set is called "purging". Moreover, 
due to autocorrelation, we have also to remove the observations that immediately follow an observation in the testing set. We call this "embargo".

### Purging the Training Set

Suppose a testing obs. with label $y_j$ is decided based on the infomration set $\Phi_j$. To prevent leakage we purge from the training data any obs. $y_i$ decided on 
the information set $\Phi_i$ whenever $\Phi_i$ overlaps with $\Phi_j$. Particularly we say there is overlap between observations $i$ and $j$ if $y_i$ and $y_j$ are concurrent.

Suppose $y_j = f([t_{j,0},t_{j,1}])$, now $y_i$ is concurrent if:

- t_{j,0}<t_{i,0}<t_{j,1}: the test window start before the training window starts but it overlaps inside
- t_{j,0}<t_{i,1}<t_{j,1}: the test window start before the training window ends but it overlaps at the other side
- t_{i,0}<t_{j,0}<t_{j,1}<t_{i,1}: one window fully contains the other

If leakage occurs performaces improve as $k$ tends to $T$ (number of bars) because the greater the number of folds, the grater the number of overlapping obs. in the
training set. 

### Embargo

If purging is not enough we can impose an embargo on training obs. after every test set. Some training labels $y_i = f([t_{i,0},t_{i,1}])$ might start after
the test interval [t_{j,0},t_{j,1}]. But, if $t_{i,0}$ is very close to $t_{j,1}$ there might be look-ahead bias. 

Thus, we introduce a small time buffer, $h$, after each test set and any trainign label starting within [t_{j,1},t_{j,1}+h] is excluded. While labels ending before the test set $t_{i,1}<t_{j,0}$ 
are fine since they only contain information available at test time.

Note:

- Usually setting $h ~ 1%$ of dataset lenght is enough
- You can check it by finding the smallest $h$ under which performance does not improve endlessly when increasing $k$












 
