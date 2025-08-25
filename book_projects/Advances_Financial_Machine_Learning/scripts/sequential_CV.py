import pandas as pd
import numpy as np
from sklearn.model_selection._split import _BaseKFold

def getTrainTimes(t1, testTimes):
    """
    Given testTimes, find the indices of training observations that do NOT overlap
    with the test intervals (i.e., purging).

    Parameters
    ----------
    t1 : pd.Series
        - index: time when the observation starts
        - value: time when the observation ends
    testTimes : pd.Series
        - index: test observation start time
        - value: test observation end time

    Returns
    -------
    trn : pd.Series
        The subset of t1 that can be used for training (after purging).
    """
    trn = t1.copy(deep=True)
    for i, j in testTimes.items():
        # training starts within test
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index  
        # training ends within test
        df1 = trn[(i <= trn) & (trn <= j)].index              
        # training interval envelops the test interval
        df2 = trn[(trn.index <= i) & (j <= trn)].index        
        trn = trn.drop(df0.union(df1).union(df2))
    return trn



def getEmbargoTimes(times: pd.DatetimeIndex, pctEmbargo: float) -> pd.Series:
    """
    Compute embargo times for each observation.
    
    Parameters
    ----------
    times : pd.DatetimeIndex
        Index of events (e.g., t1.index).
    pctEmbargo : float
        Fraction of observations to embargo (0 < pctEmbargo < 1).
    
    Returns
    -------
    pd.Series
        embargoTimes: maps each observation to the time until which 
        it should be embargoed.
    """
    n = times.shape[0]
    step = int(n * pctEmbargo)
    
    if step == 0:
        # no embargo, map each obs to itself
        mbrg = pd.Series(times, index=times)
    else:
        # embargoed: shift times forward
        mbrg = pd.Series(times[step:], index=times[:-step])
        # last 'step' obs map to last available time
        last = pd.Series(times[-1], index=times[-step:])
        # concatenate using pd.concat
        mbrg = pd.concat([mbrg, last])
    
    
    return mbrg


from sklearn.model_selection._split import _BaseKFold

class PurgedKFold(_BaseKFold):
    ''' Extend KFold class to work with labels that span intervals
        The train is purged of observations overlapping test-label intervals
        Test set is assumed contiguous (shuffle=False), w/o training samples in between
    '''
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo
    
    def split(self, X, y=None, groups=None):
        if not (X.index.equals(self.t1.index)):
            raise ValueError('X and ThruDateValues must have the same index')
        
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)  # embargo step

        # split indices into n_splits contiguous folds
        test_folds = np.array_split(np.arange(X.shape[0]), self.n_splits)
        
        for fold in test_folds:
            i, j = fold[0], fold[-1] + 1
            t0 = self.t1.index[i]  # start of test interval
            test_indices = indices[i:j]

            # max end of labels in test set
            maxT1 = self.t1.iloc[test_indices].max()
            maxT1Idx = self.t1.index.searchsorted(maxT1)

            # training indices: before test start
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)

            # append right-side train indices after embargo
            if maxT1Idx < X.shape[0]:
                train_indices = np.concatenate(
                    (train_indices, indices[maxT1Idx + mbrg:])
                )

            yield train_indices, test_indices


def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss',t1=None,cv=None,cvGen=None,pctEmbargo=None):
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score
    if cvGen is None:
        cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
    score=[]
    for train,test in cvGen.split(X=X):
        fit=clf.fit(X=X.iloc[train,:],y=y.iloc[train],sample_weight=sample_weight.iloc[train].values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X.iloc[test,:])
            score_=-log_loss(y.iloc[test],prob,sample_weight=sample_weight.iloc[test].values,labels=clf.classes_)
        else:
            pred=fit.predict(X.iloc[test,:])
            score_=accuracy_score(y.iloc[test],pred,sample_weight=sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)