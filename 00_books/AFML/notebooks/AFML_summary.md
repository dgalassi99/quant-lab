# ADVANCES IN FINANCIAL MACHINE LEARNING: RESUME

## SECTION 1 - DATA ANALYSIS

### Chapter 2: Financial Data Structures

#### Types of Financial Data

**Fundamental Data**

Mostly accounting data reported quarterly. Those have low frequency and can be used in coupling with higher frequency data.

 Typical issues?

1. Data are reported with a lapse. For example Bloomberg data are indexed by the last date included into the report, which preceeds the releasing date. Make sure to be aligned!
2. Backfilling occurs. That is filling missing data a posteriori.
3. Reinstating values. That is changing a value that wa originally mistaken. For example we have 3 valaues for GDP, the original release and two following corrections. Using the correection in an analysis were these corrections were not yet made creates errors!

**Market Data**

Trading activities taking place in an exchange. It is high frequency, not trivial to process and abundant (over 10 TB daily)

**Analytics**

Derivative data from fundamental, market or primary analytic data. So the main thing to take in consideration is that these data are pre-processed by somebody. Hence, these might be expensive and also biased!

**Alternative Data**

Data produced by individuals (social media, web searches...) or by businesses (transactions,...) or sensors (CCTV, satellites,...)

#### Bars

To apply ML we need to parse data in a regular format. We need to create tables with rows or bars. 

The goal is to transform a series of observetions at an irregular frequency into an homogeneous sample

**Time Bars**

Obtained by sampling information at fixed time intervals (timestamp, open, close, volume, ...). Even if the most popular they should be avoided.
1. Markets do not process information at a constant time interval. For example open and close hours are more active. Hnece we risk oversampling infomration when the markets are more active.
2. Exhibit poor statistical properties (heteroscedasticity, non-normality,...)

**Tick Bars**

Tick bars aggregate data after a fixed number of transactions (like 1,000 trades). This ties the analysis directly to market activity instead of time, making it a proxy for the arrival of new information in the market. Tick bars are more likely to be gaussian distributed, which is a common assumption in many ML methods.

**Volume Bars**

Tick bars can have a problem, an order of 10 shares is one tick, while 10 orders of 1 share are 10 transactions. Volume bars solve this issue by sampling every time a pre-defined number of shares. Example, sample prices everytime a contract exchanges 100 units, regardless the number of ticks involved.

**Dollar Bars**

Sampling an obs everytime a pre-defined market value is exchanged. Why?
1. If a stock appreciates 100% in X years, it will take half of stockts traded to get the same volume. Hnece, with significant price fluctuations dollar bars appear to be the best choice.
2. The number of outstanding shares changes due to corporate decisions. Dollar bars are robust agains security issues, splits and buybacks


For info on *information-driven bars* see pages 29-32

#### Multi-Product Series

If we need to model more instruments with dynamically adjustable weights or deal with products paying irregular coupons/dividends we may have issues. How can we solve it? ... the ETF trick.

##### ETF Trick

The goal is to transform any complex multi-product dataset into a single dataset that resembles a total-return ETF. 

Why is this useful? 

Because the code e can always assume that you trade
cashlike products (non-expiring cash instruments), regardless of the complexity and composition of the underlying series.

Suppose we want to develop a strategy to trade futures spread. Some issues arise.

1. Spreads have a vector of weights that changes overtime. Hence, the value of the spread might change even if prices are constant, so you can see you PnL changing because of changing weights (not prices!). You can be mislead that the stragegy is profitable when indeed only weights changed.
2. Spreads are differences or weighted combinations of prices, thus, they can assume negative values. Sometimes handling negative values can be tricky.
3. Different futures contract may have different trading hours. Thus, the spread might not be exactly tradable and you can face latency risk or slippage cause you can't trade all the legs simultaneously.
4. You need to take into account the cost of entering exiting multiple contracts along the duration of the strategy.

How do we solve it? 

Produce a time series that reflects the value of 1 USD invested in a spread. Changes in teh sereis are reflected in the PnL, the series is always postive and the other shortfall are solved. This TS is used to model, generate signal and  trade as if it was an ETF


Here is the notation:

| Symbol                    | Meaning                                         |
| ------------------------- | ----------------------------------------------- |
| $i = 1, ..., I$           | Instrument index (each contract)                |
| $t = 1, ..., T$           | Time index (bar index)                          |
| $B \subseteq \{1,...,T\}$ | Times where rebalancing happens                 |
| $\omega_{i,t}$            | Allocation weight of instrument $i$ at time $t$ |
| $o_{i,t}$, $p_{i,t}$      | Open and close prices                           |
| $\phi_{i,t}$              | USD value of one point (incl. FX)               |
| $d_{i,t}$                 | Carry, dividend, coupon, or cost                |
| $h_{i,t}$                 | Holdings of instrument $i$ at time $t$          |
| $K_t$                     | Total value of portfolio                        |
| $\delta_{i,t}$            | Price change used for PnL                       |
.

where $i = 1, ...,I$ is the instrument index and $t = 1, ...,T$ is the bar index. 

Even if some instruments are not tradable in $[t-1,t]$ they are at least tradable at $t-1$ and $t$. Take a basket of futures with allocation vector $\omega_t$ rebalanced (or rolled) on bars B in {1,..., T}.

We want to simulate how the value of $1 invested in the basket evolves over time:


1. **Holdings** $h_{i t}$: how many of each instrument we hold

$$
h_{i t} = 
\begin{cases}
\frac{\omega_{i t} K_t}{o_{i,t+1} \phi_{i t}} \Bigg/ \sum_{i=1}^I \left| \omega_{i t} \right| & \text{if } t \in B (rebalance) \\
h_{i,t-1} & \text{otherwise}
\end{cases}
$$

- when we rebalnce we recalcualte our holdings
- otherwise we keep the same holdings

2. **Price Change** $\delta_{i t}$: the price movement for each instrument 

$$
\delta_{i,t} = 
\begin{cases}
p_{i t} - o_{i t} & \text{if } (t-1) \in B \\
\Delta p_{i,t} & \text{otherwise}
\end{cases}
$$

3. **Portofolio Update** $K_{i t}$: the PnL is calculated as

$$
K_t = K_{t-1} + \sum_{i=1}^I h_{i t-1} \phi_{i t} \left( \delta_{i t} + d_{i t} \right)
$$

With $K_0 = 1$ the initial AUM. Note that dividends $d_{i t}$ are already embedded in $K_t$. 

4. **Trading Costs and Execution** ....

- **Rebalancing cost**:  

$$c_t = \sum_{i=1}^{I} \left( |h_{i,t-1}| \cdot p_{i,t} + |h_{i,t}| \cdot o_{i,t+1} \right) \cdot \phi_{i,t} \cdot \tau_i$$  

Track this cost separately. It is not embedded in $K_t$, since embedding would cause false PnL (e.g., from shorting). You can treat $c_T$ as a negative dividend** in your code.

- **Bid-ask cost**:  

$$\tilde{c}_t = \sum_{i=1}^{I} |h_{i,t-1}| \cdot p_{i,t} \cdot \phi_{i,t} \cdot \tau_i$$  

This represents the cost of crossing the spread when buying or selling one unit of the synthetic ETF.

- **Volume constraint**:  

$$v_t = \min_i \left\lbrace \frac{v_{i,t}}{|h_{i,t-1}|} \right\rbrace$$  

The least liquid instrument limits how many synthetic spread units can be traded.

##### PCA Weights

Another interesting question is: how do we calculate the vector of weights $\omega_t$?

The following method is useful to compute the portfolio allocation vector $\omega_t$ that delivers a user-defined risk distribution across the principal components of the covariance matrix $V$ (covaraince matrix of returns containing volatility and correlation)


Consider an IID Multivariate Gaussian process (like stock returns, changes in bond yields or in options volatilities) for a portfolio of N instruments) having a vector of means $\mu (Nx1)$ and a covariance matrix $V (NxN)$.

How do we proceed?

1. **Spectral decomposition**  

Decompose the covariance matrix $V$:

$$VW = W\Lambda$$

- $W$ contains eigenvectors (principal components giving us the directions of independent risk) 
- $\Lambda$ is a diagonal matrix of eigenvalues (magnitude of risk in a given direction), ordered in descending size

This operation allows us to re-express the portfolio as a combination of orthogonal (uncorrelated) risk sources.

2. **Risk of portfolio allocation $\omega$**  
   
The portfolio (total) risk is:

$$
\sigma^2 = \omega^\top V \omega = \omega^\top W \Lambda W^\top \omega = \beta^\top \Lambda \beta
$$

where:

$$
\beta = W^\top \omega
$$

represents the projection of $\omega$ in the orthogonal (principal component) basis. In other words, the vector of exposure to each component. 

3. **Risk contribution by component**  

Because $\Lambda$ is diagonal:

$$
\sigma^2 = \sum_{n=1}^N \beta_n^2 \Lambda_{n,n}
$$

Define the **risk contribution** of component $n$ as:

$$
R_n = \frac{\beta_n^2 \Lambda_{n,n}}{\sigma^2}
$$

- $\beta_n$ = exposure to component $n$

- $\Lambda_{n,n}$ = variance (risk size) of that component

- $ \beta_n^2 \Lambda_{n,n} $ = how much raw risk from component $n$

$$
\sum_{n=1}^N R_n = 1
$$

So $\{R_n\}_{n=1}^N$ is a **risk distribution** over the principal components. It tells me what % of my total portfolio risk comes from compenent $n$!

4. **Construct $\beta$ from a desired risk distribution $R$**  

Now if it is desired to allocate risk evenly or in a different wa, such as:

- Equal risk: $R_1 = R_2 = ... = R_N = \frac{1}{N}$
- Or maybe: 80% to PC1, 20% to PC2, and 0% to the rest

For this desired target risk distribution $R$, the required projection is:

$$
\beta_n = \sigma \sqrt{ \frac{R_n}{\Lambda_{n,n}} }
$$

5. **Convert $\beta$ back to allocation vector $\omega$**  

$$
\omega = W \beta
$$

Any rescaling of $\omega$ rescales $\sigma$ proportionally, but the **risk distribution $R$ remains unchanged**.

This technique is powerful because it lets you directly design how your portfolio's risk is distributed across statistically independent components (principal components) of market movement.

To summarize, PCA-based portfolio weights are exposures to assets designed to achieve a specific risk distribution across the principal components (uncorrelated risk factors) of the asset returns.

- Weights represent not simple capital allocations but the amount of exposure needed to target risk in each principal component.

- They do not sum to 1 because they are focused on risk allocation, not budget allocation.

- These weights can be positive or negative and may require normalization or scaling before being used in a real portfolio.

- The key goal is to control how risk is distributed across independent sources of variation, rather than how capital is simply split.


##### Single Future Roll

When trading futures, rolling from one contract to the next introduces price discontinuities called roll gaps. These gaps can distort the price series if not handled properly. 

**ETF Trick for Spreads**

- The ETF trick treats a spread of futures contracts as a synthetic asset. 
- This works well for multi-legged spreads and can also be applied to a single futures contract (as a 1-legged spread).
- The trick creates a continuous price series by reinvesting PnL and adjusting for roll costs, keeping the series strictly positive.

Alternatively, for one futures only...

**Roll Gaps for Single Futures Contracts**

- Compute the cumulative roll gaps.
- Subtract these cumulative gaps from the raw price series to create a continuous price series without jumps.
- Simpler and more direct for single futures.

Rolled prices are used for simulating PnL. However, raw prices should still be used to size positions and determine capital consumption. Keep in mind, rolled prices can indeed become negative, particularly in futures contracts that sold off while in contango.

In general, we wish to work with non-negative rolled series, in which case we can derive the price series of a $1 investment as follows: (1) Compute a time series of rolled futures prices, (2) compute the return (r) as rolled price change divided by the previous raw price, and (3) form a price series using those returns (i.e.,(1+r).cumprod()).

#### Sampling Features

Once a structured dataset is made how can we apply ML?

Well, we need to sample bars to produce features with relevant training examples.

##### Sampling for Reduction

Sometime downsampling the features space is useful to fit a ML algo.
Usually we do it by random sampling, tho we could miss out the most important infomration relevant to maximize the predictive power.

##### Event-Based Sampling - CUSUM Filter

Sometimes we want to spot how events affect the development of an instrument. One of the most used methods is teh CUSUM filter.

Consider IID observations $y_t$ with $t = 1,..., T$ we define the cumulative sum as:

$$
S_t = \max\left( 0,\ S_{t-1} + y_t - \mathbb{E}_{t-1}[y_t] \right)
$$

With $S_0 = 0$. A signal is generated when $S_t > h$, for some threshold value $h$.  
The filter is used to identify a sequence of positive divergences from any reset level zero.

Clearly, the concept can be extended in a symmetric case:

$$
S_{t+} = \max\left( 0,\ S_{t-1+} + y_t - \mathbb{E}_{t-1}[y_t] \right)
$$

$$
S_{t-} = \max\left( 0,\ S_{t-1-} - y_t + \mathbb{E}_{t-1}[y_t] \right)
$$

$$
S_t = \max\left( S_{t+},\ S_{t-} \right)
$$


### Chapter 3: Labelling

In this chapter we learn how to label financial data to retrieve features for supervised learning models.

#### The Fixed-Horizon Method

Consider a feture matrix $X$ with $I$ rows drawn from some bars $t = 1, ..., T$ with $T>I$. An observation $X_i$ is assigned to a label $y_i$:

- $y_i = -1$ if $r_{t_{i,0},t_{i,0}+h} < - \tau$
- $y_i = 0$ if $|$r_{t_{i,0},t_{i,0}+h}| <= \tau$
- $y_i = 1$ if $$r_{t_{i,0},t_{i,0}+h} > - \tau$

where $\tau$ is a constant threshold, $t_{i,0}$ is the index of the bar immediately after $X_i$, $t_{i,0}+h$ is the index of the h-th bar after $t_{i,0}$ and $r_{t_{i,0},t_{i,0}+h}$ the price return over an horizon $h$.

$$ r_{t_{i,0},t_{i,0}+h} = P_{t_{i,0}+h}/P_{t_{i,0}} -1 $$

Great... not really! Usually this method is applied on time bars (whihc we saw ahving poor stat. properties). Second, the same threshold $\tau$ is applied regardells the the volatility. 

Can we solve these problmes? Partially by:
- Computing a dynamic threshold $\sigma_{t_{i,0}}$ by computing a rolling exponentually weighted std of returns.
- Using volume/dollar bars
 
Why partially? Because this method does not allow us to intriduce a stop-loss (SL) or take-profit (TP) strategies. Hence, this method will result unrealistic in real operations!

#### The Triple-Barrier Method 

Label an obs. according to the first barrier it touches. We have:

- Two horizontal barriers representing the touch of TP/SL which label the obs. as +-1
- One vertical barrier touched after a given amount of elapsed bars ($h$) which labels the obs. as 0

We have to note that: (1) to label an obs. we neet to take into account the the entire path spanning $[t_{i,0},t_{i,0}+h]$; (2) we denote by $t_{i,1}$ the time of the first touch; (3) the horizontal barriers are not necessarily symmetric.

#### Size and Side

Now we want to understand how an ML algo. can learn both side and size of a trade. Ne need to elarn teh side when we do not have a model to set the sign (long/short) of our position. How do we recognize TP or SL in this situation?

We can apply the TBM direcly on all events with a fixed target...but a full pipeline consists in:

1. Apply a filter to only get some events, for example CUSUM filter
2. Calculate a dynamic target, for example as a the exponential moving volatility (getDailyVol)
3. Define a minimum return to consider an event and use a function to filter out events that do not touch this (getEvents)
4. Now you have a df with t1 indicating the timestamp of what occurs first on thpose event (tp, sl or vertical barrier hit) and teh target
5. Apply TBM to get the final labels

#### Meta-Labelling

Suppose you now have a model to set the side (long/short). We need now to learn the size of these positions.

We will discuss about this method which is used to build a secondary ML model that learns how to use a primary exogenous model.

In the original TBM:

- No side information --> the model does not know if a trade is short or long
- Horizontal barriers are symmetric
- Agnostic labelling --> +1 means that upper barrier has been hit first (TP for long/SL for short), -1 the opposite

Now we want to enhance this model by adding the side argument. And we can also set asymmetric horizontal barriers:
- ptSL[0] and [1] are the sizes of upper and lower barriers (if you set them to 0 you disable the TP/SL)
- Example:
    - For a long trade (side=+1):
        - Upper barrier = entry price + ptSl[0] * trgt (profit taking)
        - Lower barrier = entry price - ptSl[1] * trgt (stop loss)

    - For a short trade (side=-1):
        - Upper barrier = entry price - ptSl[1] * trgt (stop loss)
        - Lower barrier = entry price + ptSl[0] * trgt (profit taking)

We can update the GetEvents and GetBins functions to have an output of (0,1) to reject or confirm a trade. Basically: 

- Primary *exogenous* model (G): Generates initial trade signals (long/short), so it decides when and what direction to trade --> **side**
    - side is accepted by GetEventsMeta and GetLabelsMeta
- Secondary model (meta-labeler): Uses the labels generated by the exogenous model with side info to predict whether the primary model’s signal is likely to be profitable or not.

G produces the side series (signals) which is feed into the labelling pipeline to generate metalables (0,1) telling if the G signals were good or not. Then we can train a seocndary model to rpedict those metalables 

The secondary model’s job is to filter out bad trades by learning from historical outcomes—keeping only the signals where the expected PnL is positive (label = 1), and ignoring the others (label = 0). 

This two-stage approach can significantly improve overall strategy performance by reducing false positives from the primary model.

#### How to Use Meta-Labelling

Once we have a model to determine the side of a bet we need one to determine the size. That is ... *How much money I am willing to bet in this position?*

We have seen that meta-labelling covers this specific question. *When should we specifically use it?*

In binary classification problems we deal with the trade-off between FP and FN errors. The goal is to increase the TP rate but also FP rate will increase.

Definitions:
- FP: positive instances incorrectly classified positive
- FN: negative instances incorrectly classified negative
- TP: positive instances correctly classified positive
- TN: negative instances correctly classified negative
- TPR: True Positive Rate --> positive instances correctly classified/positive instances $TP/(TP+FN)$ --> proportion of positive instances that were correctly classified as positive by the model
- FPR: False Negative Rate --> negative instances incorrecly classfied as positive/negative instances $FP/(FP+TN)$ --> proportion of positive instances that are inaccurately detected as positive
- Precision: $TP/(TP+FP)$ --> correctly classified positive/all classfied as positive
- Recall (TPR): $TP/(TP+FN)$ --> correctly classified positive/number of real positives
- Accuracy: $(TP+TN)/(TP+TN+FN+FP)$ --> correct classficiations/total classifications
- F1 Score = $2/(1/Precision+1/Recall)$
The trade off between TPR and FPR is the key problem of binary classification (usually represented on the ROC curve).

This if we aim at improving the precision by lowering FP, this results in an increase of FN (decrease of Recall/TPR) and viceversa. So the usual metric we want to maximie is their harmonic average: F1 Score. Meta-labelling helps us in achieving this result... *how?*

1. Build a model with high recall and low precision (low FN) --> "do not lose opportunities logic"
2. Apply meta-labelling to the positive predicted by the primary model --> "filter out to keep only good opportunities"

### Chapter 4: Sampling Weights

In this chapter we learn how to deal with the fact that financial observations are not IID.

#### Overlapping Outcomes

Given a observed feature $X_i$ we can assign a label $y_i$ function of bars occurred between $t_{i,0}$ and $t_{i,1}$. Now suppose that $t_{i,1} > t_{i,0}$ then $y_i$ and $y_j$ depend on a common return. 

For example:
| Label | Interval      | Start $t_0$       | End $t_1$         | Return |
| ----- | ------------- | ----------------- | ----------------- | ------ |
| $y_1$ | 09:00 → 09:02 | $t_{1,0} = 09:00$ | $t_{1,1} = 09:02$ | 0.02   |
| $y_2$ | 09:01 → 09:03 | $t_{2,0} = 09:01$ | $t_{2,1} = 09:03$ | 0.0198 |

Then the series of labels are not IID. Now, we could resctrict the intervals horizon to prevent this, but this reduces our horizon to the frequency of the observations itself... not a great idea.

*How to we tackle the non IIDness of financial observations?*

#### Number of Concurrent Labels

A label $y_i$ is a function of the returns in its interval $[t_{i,0} , t_{i,1}]$. Now, a return $r_{t-1,t}$ is the return occurring between price points $p_{t-1}$ and $p_t$. We say that labels are concurrent in $t$ if they include this return in their time intervals.

Define for each $t$ and $y_i$ a binary array $1_{t,i}$. This is 1 if $[t_{i,0} , t_{i,1}]$ overlaps $[t-1, t]$ and 0 otherwise. We can then compute the number of concurrent labels at $t$ as:

$$
c_t = \sum_{i=1}^I 1_{t,i} 
$$

#### Average Uniqueness of a Label

Given a label $y_i$ at time $t$ we can define:
- Uniqueness of a label: $u_{t,i} = 1_{t,i} c_t^{-1}$
- Average uniqueness of a label: $u_i$ = $\frac{\sum_{t=1}^T u_{t,i}}{\sum_{t=1}^T 1_{t,i}}$

Generally speaking, we expect $u_i <1$ and the more it gets close to 1 the more the label is unique (no-overlap). It must be noted that, calcualting the average uniqueness requires infomration about the future, but this is not an issue as $u_i$ is only used on the training set, hence there is no data leakage.

#### Bagging Classifiers and Uniqueness 

Suppose we drawn with replacement (boostrapping) observations. Now after $I$ drawns the probability of picking $i$ os $(1-I^-1)^I$. As the sample size grows this probability converges to $1/e$. Hnece, the number of unique observation draws is expected to be $1-1/e ~ 2/3$.

Now, if the maximum number of non-overlapping outcomes is $K<I$ we can't pick $I$ times because of overlap, thus the number of unique obs. drawn is $1-e^{-K/I} < 1-e^{-1}$. This means that assuming IID leads to oversampling!

When $I^{-1} \sum_{i=1}^I u_i << 1$ (that is there is a lot of overlaping) it becomes increasingly likely that in-bag observations will be 
1. Redundant
2. Very similar to OOB obs. (we treat this in chap. 7)

Making the boostrap inefficient. For example, in Random Forest we will have a lot of similar and overfitted trees.

What do we do? 

1. Drop overlapping outcomes. This, tho, results in data/information loss.
2. Use the average uniqueness information to reduce the influence of outcomes containing redundant info by only sampling a fraction of obs out['tW'].mean() or a multiple of this (can be used in max_samples arguemnt in sklearn.ensemble.BaggingClassifier). This way IB obs. are not sampled at frequencies higher than their uniqueness.
3. Use sequential boostrap...

##### Sequential Bootstrap

In sequantial bootstrap drawns are made according to a change in probability that controls for redundancy. An observation $X_i$ is picked from a uniform distribution $i$ ~ $U[1,I]$. That is the probability to be picked is $\delta_i^{(1)} = I^{-1}$ and overlap among labels are ingored.

Now, ww proceed like this:
1. Pick the first obs with probability $\delta_i^{(1)} = I^{-1}$ and denote by $\phi^{(1)} = {i}$ the sequence of chosen samples so far
2. For the next candidate $j$, we measure its uniqueness relative to what's already drawn (if j overlaps a lot to the chosen lables the denominator increases and the uniqueness shrinks)

$$ u_{t,j}^{(2)} = \frac{1_{t,j}}{1+\sum_{k \in \mathbf{\phi^{(1)}}} 1_{t,k}} $$

3. Then the average uniquenss over the lifespan of j (scores how unique is label $j$ at this step) is:

$$ u_j^{(2)} = \frac{\sum_{t=1}^T u_{t,j}}{\sum_{t=1}^T 1_{t,j}} $$

4. Finally, update the likelihood of the second drawn to reduce the chance of picking overlapped labels:

$$ \delta_j^{(2)} = \frac{u_j^{(2)}}{\sum_{k=1}^I u_k^{(2)}} $$

Note that those $\delta_j^{(2)}$ are scaled to sum to 1. Now we can do another drawn, update $\phi^{(2)}$ and re-evaluate $\delta_j^{(3)}$. This process continues until $I$ drawns are made.

The benefits are that:
- Overlaps/repetitions are still possible
- Overlaps/repetitions are increasingly less likely
- Teh bootrap sample will be close to an IID sample 

## Section 5


### Chapter 20: Multiprocessing and Vectorization

A process is a fully independent program with its own memory. Hence, multiprocessing means running several processes that do not share memory. In PY this is the best way to achieve true parallelization.
