# DATA ANALYSIS - FRACTIONALLY DIFFERENCIATED FEATURES

### Stationarity vs Memory Dilemma

Let's first clarify that:

- Stationary series (such are log differences of prices or returns): have mean, variance and statistical properties that do not change overtime, but do not have memory
- Non-stationary series (such as prices): have non constant stat. properties and "store" memory as the next value in the series is strongly influenced by the previous one

Generally, for statistics and ML we need stationarity as the model need to generalize on unseed data, but for a good prediction we also need to save some memory otherwise there is no forecast.

Hence, the dilemma is... *what should we prioritize? What is the minimum amount of differentiation to be applied to a price series so that it becomes statioanry but conserves most of the memory?*

- No differencing --> raw prices --> full memory but fully non-stationary
- Standard differencing --> returns --> stationary but too aggressive as it fully kills the memory
- Fractional Differencing --> the trade-off we are looking for
 
## The Method

We consider a backshift oberator $B^k$ such that when applied to a matrix/series it shifts it back of k-periods. That is $B^kX_t = X_{t-k}$. Simply saying the power of $B$ indicates how many periods we have to lag. For, example 
The standdard differencing process (to calculate returns) is written as $(1-B)X_t = X_t - BX_t = X_t - X_{t-1}$, if we square the operator $(1-B)^2 = X_t - 2X_{t-1} - X_{t-2}$ and so on by increasing the power. In these processes integer powers
are usually used. But what if we allow this exponent to be fractional (a real number such as 0.3, 0.5, ...)?

$$
(1 - B)^d = \sum_{k=0}^\infty \binom{d}{k} (-B)^k = \sum_{k=0}^\infty \(-B)^k \prod_{i=0}^{k-1}{\frac{d-1}{k-1}} = 1-dB + \frac{d(d-1)B^2}{2!} - \frac{d(d-1)(d-2)B^3}{3!} + ...
$$

### Long Memory

Now we can write this as:

$$
Z_t = \sum_{k=0}^\infty \omega_k X_{t-k}
$$

Where the values $X = [X_t, X_{t-1}, ..., X_{t-k}, ...]$ and weights vector $\omega$ such that $\omega_k = (-1)^k \frac{\prod_{i=0}^{k-1} (d-i)}{k!}$. Hence, for example:

- $\omega_0 = 1, \quad$
- $\omega_1 = -d, \quad$
- $\omega_2 = \frac{d(d-1)}{2!}, \quad$
- $\omega_3 = -\frac{d(d-1)(d-2)}{3!}, \quad \dots$

Now when $d$ is an integer the term $\frac{\prod_{i=0}^{k-1} (d-i)}{k!} = 0$ and memory is cancelled when $d<k$. As an example with $d=1$ the weight look like $\omega = [1,-1,0,0,...]$.


  
