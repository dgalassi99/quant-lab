# DATA ANALYSIS - SAMPLING WEIGHT

In this chapter we learn how to deal with the fact that financial observations are not IID.

## Overlapping Outcomes

Given a observed feature $X_i$ we can assign a label $y_i$ function of bars occurred between $t_{i,0}$ and $t_{i,1}$. Now suppose that $t_{i,1} > t_{i,0}$ then $y_i$ and $y_j$ depend on a common return. 

For example:
| Label | Interval      | Start $t_0$       | End $t_1$         | Return |
| ----- | ------------- | ----------------- | ----------------- | ------ |
| $y_1$ | 09:00 → 09:02 | $t_{1,0} = 09:00$ | $t_{1,1} = 09:02$ | 0.02   |
| $y_2$ | 09:01 → 09:03 | $t_{2,0} = 09:01$ | $t_{2,1} = 09:03$ | 0.0198 |

Then the series of labels are not IID. Now, we could resctrict the intervals horizon to prevent this, but this reduces our horizon to the frequency of the observations itself... not a great idea.

*How to we tackle the non IIDness of financial observations?*

## Number of Concurrent Labels

A label $y_i$ is a function of the returns in its interval $[t_{i,0} , t_{i,1}]$. Now, a return $r_{t-1,t}$ is the return occurring between price points $p_{t-1}$ and $p_t$. We say that labels are concurrent in $t$ if they include this return in their time intervals.

Define for each $t$ and $y_i$ a binary array $1_{t,i}$. This is 1 if $[t_{i,0} , t_{i,1}]$ overlaps $[t-1, t]$ and 0 otherwise. We can then compute the number of concurrent labels at $t$ as:

$$
c_t = \sum_{i=1}^I 1_{t,i} 
$$

## Average Uniqueness of a Label

Given a label $y_i$ at time $t$ we can define:
- Uniqueness of a label: $u_{t,i} = 1_{t,i} c_t^{-1}$
- Average uniqueness of a label: $u_i$ = $\frac{\sum_{t=1}^T u_{t,i}}{\sum_{t=1}^T 1_{t,i}}$

Generally speaking, we expect $u_i <1$ and the more it gets close to 1 the more the label is unique (no-overlap). It must be noted that, calcualting the average uniqueness requires infomration about the future, but this is not an issue as $u_i$ is only used on the training set, hence there is no data leakage.

## Bagging Classifiers and Uniqueness 

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

### Sequential Bootstrap

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
- The bootrap sample will be close to an IID sample 

### Monte Carlo Experiments 

We can check the sequantial bootsrap by generating random label intervals. Each observation starts at a random bar and spand a random number of bars (we can set a max number of bars maxH). Once we have this randomly generated t1 we can test the average uniqueness of the labels when using a classing sampling with replacement bootstrap or our sequential bootstrap 

## Return Attribution

We will now disucss how to assign weights to observations in order to take into account that: (1) overlapping outcomes are overweighted compared to non-overlapping ones; (2) labels associated with higher absolute returns should weight more.

When labels are fucntion of the return sign ({-1,1} for standard labelling {0,1} for meta-labelling) the sample weights are $v_i$. Summing per bar return over the event lifespan and correcting for concurrency. Then we normalize $v_i$ to get $w_i$ whicha re summing to 1.

$$ v_i = \left| \sum_{t = t_{i,0}}^{t_{i,1}} \frac{r_{t-1,t}}{c_t} \right| $$

$$ w_i = \frac{v_i}{\sum_{i=1}^{I} v_i} $$

The idea is to weight an observation as a function of the absolute log return that can uniquely be attributed to it. Using a weight inverse to the return would overweight low-return events whihc we indeed filter out as we have ssen in the previous chapters.

## Time Decay

As markets adapt we might want to underweight older observation. To do it we will adjust our weights by a decay vector $d(x)$ which depends on the cumulative uniquenss of $x$ of the events. This measures how much an event's "signal" is unique relative to other overlapping events. The decay is applied along uniquenss and not strinctly on chronological time.

$$ x \in \left[0, \sum_{i=1}^{I} u_i\right] $$

We aim at a linear piecewise decay function of hte form: $d(x) = \max{[0,a+bx]}$

Case (a): $c \in [0,1]$  *(light decay for older events)*

The user sets $c = d[1]$ the minimum relative weight of the first event compared to the last event. We solve for slope $b$ and intercept $a$:

1. Final weight condition: $a + b \sum_{i=1}^I u_i = 1$ which implies $a = 1 - b \sum_{i=1}^I u_i$

2. First weight condition: $a + b \cdot 0 = c$ which implies $b = \frac{1-c}{\sum_{i=1}^I u_i}$

Case (b): $c \in (0,-1)$  *(heavy decay - some events assingd to zero)*

Here the user specifies a negative $c$, so that some older events are completely ignored.  

1. Condition at cutofff: $a - b c \sum_{i=1}^I u_i = 0$ which implies $b = \frac{1+c}{\sum_{i=1}^I u_i}$

2. From final weight condition: $a = 1 - b \sum_{i=1}^I u_i$

This is a sort of double correction on weights by uniqueness and by time. Let's visualize an example.

Say we have 5 events with uniqueness weights as following and we apply a linear decay factor assuming cumulative uniqueness is simply the cumulative sum of the weights:

| Event | $w_i$ | x (cumsum) |
| ----- | ----- | ---------- |
| 1     | 0.2   | 0.2        |
| 2     | 0.3   | 0.5        |
| 3     | 0.1   | 0.6        |
| 4     | 0.25  | 0.85       |
| 5     | 0.15  | 1.0        |

Now we pick a light decay parameter $c = 0.5$ and find $a$ and $b$:

| Event | x    | $d[x] = \max(0, 0.5 + 0.5 x)$ |
| ----- | ---- | ----------------------------- |
| 1     | 0.2  | 0.5 + 0.5\*0.2 = 0.6          |
| 2     | 0.5  | 0.5 + 0.5\*0.5 = 0.75         |
| 3     | 0.6  | 0.5 + 0.5\*0.6 = 0.8          |
| 4     | 0.85 | 0.5 + 0.5\*0.85 = 0.925       |
| 5     | 1.0  | 0.5 + 0.5\*1 = 1.0            |

Then the final weights are:

| Event | $w_i$ | $d[x]$ | $w_i^{\text{final}}$ |
| ----- | ----- | ------ | -------------------- |
| 1     | 0.2   | 0.6    | 0.12                 |
| 2     | 0.3   | 0.75   | 0.225                |
| 3     | 0.1   | 0.8    | 0.08                 |
| 4     | 0.25  | 0.925  | 0.23125              |
| 5     | 0.15  | 1.0    | 0.15                 |

We observe that:

- We want to down-weight redundant labels, because too many overlapping events bias the sample.
- If we only use uniqueness as weights, that handles redundancy locally (event by event).
- Time-decay adds another correction: “older” information should be discounted relative to the stream of unique information coming in.

Why cumulative uniquenss and not just time?
- If we used plain chronological time: early events are penalized just for being early, even if they were unique.
- With cumulative uniqueness: we think in terms of how much fresh unique information has entered the system so far.
- We’re treating “age” not in terms of the clock, but in terms of the information budget already consumed. It’s like saying:
“I don’t care if this observation happened 2 hours ago. What matters is: how much new, non-redundant signal have we seen since then?

It is worth to note some special cases:
- $c = 1$ means no decay
- $0<c<1$ means that weights decay linearly, but all observations will have a weights different than 0
- $c=0$ means that weights decay linearly to zero
- $c<0$ means that the oldest portion of obs. are eliminated (weights are set to zero)

## Class Weights

While sample weights adjust the importance of each observation, class weights adjust the importance of each label (class). Say we have a case in which the data is unbalanced where 99.9% of times is +1 (normal days) and 0.1% is -1 (market crash). The model will erach 99.9% accuracy just by always predictiong =1, but it will never spot a financial crisis. Hence, we need to increase the class weight of the rare events (improving recall).

- Pass class_weight = 'balanced' which sets class weights inversely proportional to class frequency
- Pass class_weight = 'balanced_subsample' whihc does the same but inside each bootstrap sample rahter than in the entire dataset
