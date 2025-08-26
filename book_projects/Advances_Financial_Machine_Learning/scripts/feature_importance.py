import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from sequential_CV import PurgedKFold, cvScore
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# 1. MDI: Mean Decrease Impurity
# -------------------------------------------------------------------
def featImpMDI(fit, featNames):
    """
    Feature importance based on in-sample mean impurity reduction (MDI).
    fit: fitted BaggingClassifier or RandomForest
    featNames: list of feature names
    """
    # Collect feature importances from each tree
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index', columns=featNames)
    df0 = df0.replace(0, np.nan)  # avoid zeros (since max_features=1)
    
    imp = pd.concat({
        'mean': df0.mean(),
        'std': df0.std() * df0.shape[0] ** -0.5  # standard error
    }, axis=1)
    
    # Normalize to sum to 1
    imp['mean'] /= imp['mean'].sum()
    return imp


# -------------------------------------------------------------------
# 2. MDA: Mean Decrease Accuracy
# -------------------------------------------------------------------
def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring='neg_log_loss'):
    """
    Feature importance based on out-of-sample score reduction (MDA).
    clf: sklearn classifier
    X, y: features and labels
    cv: number of folds
    sample_weight: Series of sample weights
    t1: Series of event end times (for PurgedKFold)
    pctEmbargo: embargo percentage
    scoring: 'neg_log_loss' or 'accuracy'
    """
    
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise ValueError("scoring must be either 'neg_log_loss' or 'accuracy'")
    
    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    scr0 = pd.Series(dtype=float)                 # base scores
    scr1 = pd.DataFrame(columns=X.columns, dtype=float)  # shuffled scores
    
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)
        
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1.values,
                                    labels=clf.classes_)
        else:  # accuracy
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)
        
        # Permute each feature one at a time
        for j in X.columns:
            X1_ = X1.copy()
            X1_.loc[:, j] = np.random.permutation(X1_[j].values)  # safer than shuffle in place
            
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob, sample_weight=w1.values,
                                           labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1.values)
    
    # Importance = baseline - permuted
    imp = (-scr1).add(scr0, axis=0)
    if scoring == 'neg_log_loss':
        imp = imp / -scr1
    else:
        imp = imp / (1.0 - scr1)
    
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** -0.5}, axis=1)
    return imp, scr0.mean()


# -------------------------------------------------------------------
# 3. SFI: Single Feature Importance
# -------------------------------------------------------------------
def auxFeatImpSFI(featNames, clf, trnsX, cont, scoring, cvGen):
    """
    Single Feature Importance (SFI).
    Trains the classifier on one feature at a time.
    """
    imp = pd.DataFrame(columns=['mean', 'std'])
    
    for featName in featNames:
        df0 = cvScore(clf, X=trnsX[[featName]], y=cont['bin'],
                      sample_weight=cont['w'], scoring=scoring, cvGen=cvGen)
        imp.loc[featName, 'mean'] = df0.mean()
        imp.loc[featName, 'std'] = df0.std() * df0.shape[0] ** -0.5
    
    return imp



# -------------------------------------------------------------------
# 4. CREATE A SYNTHETIC DATASET
# -------------------------------------------------------------------

def getTestData(n_features, n_informative, n_redundant, n_samples):
    '''"""
    Generate a synthetic classification dataset with financial-style indexing.

    Parameters
    ----------
    n_features : int Total number of features.
    n_informative : int Number of informative features.
    n_redundant : int Number of redundant features.
    n_samples : int Number of samples (rows).

    Returns
    -------
    X : pd.DataFrame Features with business-day index.
    y : pd.DataFrame Labels with weights and 't1' column (end time = itself).
    """
    '''
    #generate classification problem
    X, y = make_classification(n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=0,
        shuffle=False)
    #business-day datetime index ending today
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_samples, freq="B")
    #cnvert X and y to DataFrames
    X = pd.DataFrame(X, index=dates)
    y = pd.Series(y, index=dates, name="bin").to_frame()
    #feature names: Informative (I), Redundant (R), Noise (N)
    feature_names = (
        [f"I_{i}" for i in range(n_informative)]
        + [f"R_{i}" for i in range(n_redundant)]
        + [f"N_{i}" for i in range(n_features - (n_informative + n_redundant))])
    #assign names to feature matrix
    X.columns = feature_names
    #add sample weights (uniform here)
    y["w"] = 1.0 / len(y)
    #add t1 (end time = itself, synthetic case)
    y["t1"] = y.index

    return X, y



# -------------------------------------------------------------------
# 5. FEATURE IMPORTANCE FOR ANY METHOD
# -------------------------------------------------------------------

def featImportance(trnsX, cont, feature_masking = False, n_estimators=1000, cv=10, max_samples=1.,
                   numThreads=1, pctEmbargo=0, scoring='accuracy',
                   method='SFI', minWLeaf=0., random_state=42, **kargs):
    """
    Wrapper to compute feature importance using:
    - MDI (Mean Decrease Impurity)
    - MDA (Mean Decrease Accuracy)
    - SFI (Single Feature Importance)

    Parameters
    ----------
    trnsX : pd.DataFrame Features
    cont : pd.DataFrame Contains 'bin' (labels), 'w' (weights), 't1' (event end times)
    """
    if feature_masking:
        max_features = 'sqrt'
    else:
        max_features = 1
    # ---------------------------
    # 1) Classifier definition
    # ---------------------------
    base_tree = DecisionTreeClassifier(
        criterion='entropy',
        max_features=1, # --> prevent masking
        class_weight='balanced',
        min_weight_fraction_leaf=minWLeaf,
        random_state=random_state
    )

    clf = BaggingClassifier(
        estimator=base_tree, # --> modern API
        n_estimators=n_estimators,
        max_features=1.,
        max_samples=max_samples,
        oob_score=True,
        n_jobs=-1 if numThreads > 1 else 1,
        random_state=random_state
    )

    #Fit once on all data (needed for OOB and MDI)
    fit = clf.fit(X=trnsX, y=cont['bin'], sample_weight=cont['w'].values)
    oob = fit.oob_score_

    # ---------------------------
    # 2) Choose method
    # ---------------------------
    if method == 'MDI':
        imp = featImpMDI(fit, featNames=trnsX.columns)
        oos = cvScore(
            clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'],
            t1=cont['t1'], pctEmbargo=pctEmbargo, scoring=scoring
        ).mean()

    elif method == 'MDA':
        imp, oos = featImpMDA(
            clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'],
            t1=cont['t1'], pctEmbargo=pctEmbargo, scoring=scoring
        )

    elif method == 'SFI':
        cvGen = PurgedKFold(n_splits=cv, t1=cont['t1'], pctEmbargo=pctEmbargo)
        oos = cvScore(
            clf, X=trnsX, y=cont['bin'], sample_weight=cont['w'],
            scoring=scoring, cvGen=cvGen
        ).mean()

        # Parallelize feature-wise loop with joblib
        def _process_feat(feat):
            return feat, auxFeatImpSFI([feat], clf, trnsX, cont, scoring, cvGen)

        results = Parallel(n_jobs=numThreads)(
            delayed(_process_feat)(feat) for feat in trnsX.columns
        )

        # Collect into DataFrame
        imp = pd.DataFrame({feat: res.loc[feat] for feat, res in results}).T

    else:
        raise ValueError("method must be one of ['MDI', 'MDA', 'SFI']")

    return imp, oob, oos

# -------------------------------------------------------------------
# 6. AUTOMATED FULL PIPELINE
# -------------------------------------------------------------------

import pandas as pd
from itertools import product

def testFunc(n_features=40, n_informative=10, n_redundant=10, n_estimators=1000,
             n_samples=10000, cv=10, save_csv = False, pathOut="./testFunc/"):
    """
    Test the performance of the feature importance functions (MDI, MDA, SFI)
    on artificial data.
    """

    # --- 1) Synthetic dataset ---
    trnsX, cont = getTestData(
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_samples=n_samples
    )

    # --- 2) Grid of configs ---
    param_grid = {
        "minWLeaf": [0.],
        "scoring": ["accuracy"],
        "method": ["MDI", "MDA", "SFI"],
        "max_samples": [1.]
    }

    jobs = []
    for vals in product(*param_grid.values()):
        jobs.append(dict(zip(param_grid.keys(), vals)))

    results = []
    kargs = {
        "pathOut": pathOut,
        "n_estimators": n_estimators,
        "tag": "testFunc",
        "cv": cv
    }

    # --- 3) Loop over configs ---
    for job in jobs:
        job["simNum"] = (
            f"{job['method']}_{job['scoring']}_"
            f"{job['minWLeaf']:.2f}_{job['max_samples']}"
        )
        print("Running:", job["simNum"])

        # merge with base args
        run_args = {**kargs, **job}

        # compute FI
        imp, oob, oos = featImportance(trnsX=trnsX, cont=cont, **run_args)

        # optional plotting (if you’ve got a plotting util)
        # plotFeatImportance(imp=imp, oob=oob, oos=oos, **run_args)

        # --- summarize results ---
        df0 = imp[['mean']] / imp['mean'].abs().sum()
        df0['type'] = [i[0] for i in df0.index]   # feature type (I, R, N)
        df0 = df0.groupby('type')['mean'].sum().to_dict()
        df0.update({"oob": oob, "oos": oos})
        df0.update(job)
        results.append(df0)

    # --- 4) Collect + save ---
    out = pd.DataFrame(results).sort_values(
        ["method", "scoring", "minWLeaf", "max_samples"]
    )
    out = out[["method", "scoring", "minWLeaf", "max_samples",
               "I", "R", "N", "oob", "oos"]]
    if save_csv:
        out.to_csv(f"{pathOut}/stats.csv", index=False)

    return out

# -------------------------------------------------------------------
# 7. PLOTTING
# -------------------------------------------------------------------


def plotFeatImportance(pathOut, imp, oob, oos, method, tag=0, simNum=0, plot_to_save = False, **kargs):
    """
    Plot feature importances with error bars.

    Parameters
    ----------
    pathOut : str
        Output folder for saving the plot.
    imp : pd.DataFrame
        Feature importance DataFrame with 'mean' and 'std' columns.
    oob : float
        Out-of-bag score.
    oos : float
        Out-of-sample score.
    method : str
        Importance method ('MDI', 'MDA', 'SFI').
    tag : str/int
        Experiment tag for labeling.
    simNum : str/int
        Simulation ID for labeling.
    plot_to_save : bool
    If True, save plot to pathOut; if False, display inline only
    """

    # sort by mean importance
    imp_sorted = imp.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, imp_sorted.shape[0] / 5)))

    # horizontal bar plot with error bars
    ax.barh(
        imp_sorted.index,
        imp_sorted["mean"],
        xerr=imp_sorted["std"],
        color="skyblue",
        alpha=0.6,
        ecolor="red"
    )

    # MDI baseline line (random expectation)
    if method == "MDI":
        ax.axvline(1.0 / imp.shape[0], color="red", linestyle="dotted", linewidth=1)

    oob_str = f"{oob:.4f}" if oob is not None else "N/A"
    oos_str = f"{oos:.4f}" if oos is not None else "N/A"
    
    # title with performance metrics
    ax.set_title(
        f"Feature Importance ({method})\n"
        f"tag={tag} | simNum={simNum} | oob={oob_str} | oos={oos_str}",
        fontsize=12
    )

    # clean layout
    ax.set_xlabel("Importance")
    plt.tight_layout()

    if plot_to_save:
        out_path = f"{pathOut}/featImportance_{simNum}.png"
        plt.savefig(out_path, dpi=100)
        plt.close(fig)
        return out_path
    else:
        plt.show()
        return None

# -------------------------------------------------------------------
# 8. GET EIGVAL/EIGVEC
# -------------------------------------------------------------------


def get_eVec(dot: pd.DataFrame, varThres: float = 0.95):
    """
    Compute eigenvectors from a dot-product matrix and reduce dimensionality
    based on a cumulative variance threshold.

    Parameters
    ----------
    dot : pd.DataFrame
        Feature covariance-like dot-product matrix (features x features)
    varThres : float
        Fraction of total variance to retain (0 < varThres <= 1)

    Returns
    -------
    eVal : pd.Series
        Sorted eigenvalues (descending) of selected principal components
    eVec : pd.DataFrame
        Corresponding eigenvectors (columns = PCs, rows = original features)
    """
    # Eigen decomposition
    eVal, eVec = np.linalg.eigh(dot)
    
    # Sort eigenvalues/eigenvectors in descending order
    idx = eVal.argsort()[::-1]
    eVal, eVec = eVal[idx], eVec[:, idx]
    
    # Convert to pandas for convenience
    eVal = pd.Series(eVal, index=[f'PC_{i+1}' for i in range(len(eVal))])
    eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)
    
    # Retain top PCs based on cumulative variance
    cumVar = eVal.cumsum() / eVal.sum()
    dim = cumVar.values.searchsorted(varThres)
    eVal = eVal.iloc[:dim+1]
    eVec = eVec.iloc[:, :dim+1]
    
    return eVal, eVec

# -------------------------------------------------------------------
# 8. APPLY PCA
# -------------------------------------------------------------------

def orthoFeats(dfX: pd.DataFrame, varThres: float = 0.95) -> pd.DataFrame:
    """
    Transform features into orthogonal components (principal components),
    reducing dimensionality while retaining a specified fraction of variance.

    Parameters
    ----------
    dfX : pd.DataFrame
        Original feature matrix (observations x features)
    varThres : float
        Fraction of variance to retain for dimensionality reduction

    Returns
    -------
    dfP : np.ndarray
        Transformed orthogonal features (observations x reduced features)
    """
    # Standardize features
    dfZ = (dfX - dfX.mean()) / dfX.std()
    
    # Compute dot-product (covariance-like) matrix
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    
    # Eigen decomposition and dimensionality reduction
    eVal, eVec = get_eVec(dot, varThres)
    
    # Project standardized features onto selected eigenvectors
    dfP = np.dot(dfZ, eVec)
    
    return pd.DataFrame(dfP, index=dfX.index, columns=eVec.columns)