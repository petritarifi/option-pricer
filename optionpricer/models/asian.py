"""
Asian (average-price) options on a discretely monitored average.

The average is taken over n_fixings equally spaced dates t_i = i*T/n_fixings,
i = 1..n_fixings (the spot at t = 0 is not a fixing), and the option pays
max(A - K, 0) or max(K - A, 0) at T.

- geometric_asian_price : exact closed form for the geometric average.
- levy_asian_price      : lognormal two-moment approximation for the
                          arithmetic average (Turnbull-Wakeman / Levy).
- mc_asian_price        : Monte Carlo for the arithmetic average, with
                          antithetic variates and the geometric Asian as
                          a control variate.
"""
import numpy as np

from optionpricer.models.black_scholes import black76_price


def _check_option_type(option_type):
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type has to be 'call' or 'put', received: {option_type}")


def _payoff(average, K, option_type):
    if option_type == "call":
        return np.maximum(average - K, 0.0)
    return np.maximum(K - average, 0.0)


def geometric_asian_price(S0, K, r, T, vol, q, option_type, n_fixings):
    """
    Closed-form price of a discretely monitored geometric-average Asian option.

    Under GBM, log G = (1/n) * sum(log S_ti) is a sum of Gaussians, so G is
    exactly lognormal with
        E[log G]   = log S0 + (r - q - vol^2/2) * T * (n+1) / (2n)
        Var[log G] = vol^2 * T * (n+1) * (2n+1) / (6n^2)
    and the option is a Black-76 option on the forward E[G].
    """
    _check_option_type(option_type)
    n = n_fixings
    mean_log = np.log(S0) + (r - q - 0.5*vol**2) * T * (n + 1) / (2*n)
    var_log = vol**2 * T * (n + 1) * (2*n + 1) / (6 * n**2)

    forward_G = np.exp(mean_log + 0.5*var_log)
    return black76_price(forward_G, K, r, T, np.sqrt(var_log / T), option_type)


def arithmetic_average_moments(S0, r, T, vol, q, n_fixings):
    """
    Exact first two moments of the discrete arithmetic average A under GBM.

        E[A]   = (1/n)   * sum_i     S0 * exp((r-q) t_i)
        E[A^2] = (1/n^2) * sum_i,j   S0^2 * exp((r-q)(t_i + t_j) + vol^2 min(t_i, t_j))
    """
    t = T * np.arange(1, n_fixings + 1) / n_fixings
    g = r - q
    m1 = S0 * np.mean(np.exp(g * t))
    m2 = S0**2 * np.mean(np.exp(g * np.add.outer(t, t) + vol**2 * np.minimum.outer(t, t)))
    return m1, m2


def levy_asian_price(S0, K, r, T, vol, q, option_type, n_fixings):
    """
    Arithmetic-average Asian option by lognormal moment matching.

    The arithmetic average is not lognormal, but it is approximated by the
    lognormal variable with the same first two moments (Turnbull & Wakeman
    1991, Levy 1992), which is then priced with Black-76:
        F_A     = E[A]
        vol_A^2 = log(E[A^2] / E[A]^2) / T

    The moments are the exact ones for the discrete average. This is an
    approximation, independent of the Monte Carlo pricer: accurate for low
    vol * sqrt(T), with an error that grows as the average's distribution
    becomes more skewed than a lognormal.
    """
    _check_option_type(option_type)
    m1, m2 = arithmetic_average_moments(S0, r, T, vol, q, n_fixings)
    vol_A = np.sqrt(np.log(m2 / m1**2) / T)
    return black76_price(m1, K, r, T, vol_A, option_type)


def simulate_averages(S0, r, T, vol, q, n_fixings, n_paths, antithetic=True, seed=None):
    """
    Simulate GBM paths on the fixing grid and return (arithmetic, geometric)
    averages, one per path.

    Paths are stepped exactly (log-Euler is exact for GBM) and the averages
    are accumulated on the fly, so memory is O(n_paths) rather than
    O(n_paths * n_fixings). With antithetic=True, path i and path i + half
    are driven by opposite Gaussian increments.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_fixings
    drift = (r - q - 0.5*vol**2) * dt
    diffusion = vol * np.sqrt(dt)

    n_draws = n_paths // 2 if antithetic else n_paths
    n_sim = 2 * n_draws if antithetic else n_draws

    log_S = np.full(n_sim, np.log(S0))
    sum_S = np.zeros(n_sim)
    sum_log_S = np.zeros(n_sim)

    for _ in range(n_fixings):
        Z = rng.standard_normal(n_draws)
        if antithetic:
            Z = np.concatenate([Z, -Z])
        log_S += drift + diffusion * Z
        sum_S += np.exp(log_S)
        sum_log_S += log_S

    return sum_S / n_fixings, np.exp(sum_log_S / n_fixings)


def mc_asian_price(S0, K, r, T, vol, q, option_type, n_fixings, n_paths,
                   antithetic=True, control_variate=True, seed=None):
    """
    Arithmetic-average Asian option price by Monte Carlo.
    Returns (price, stderr) — use ~1.96*stderr for a 95% CI.

    control_variate=True uses the geometric Asian on the same paths, whose
    price is known in closed form. The two payoffs are almost perfectly
    correlated, so the regression-adjusted estimator
        Y - b * (X - E[X]),   b = Cov(Y, X) / Var(X)
    removes most of the variance of Y. b is estimated from the same sample,
    which adds a bias of order 1/n_paths, negligible next to the stderr.
    """
    _check_option_type(option_type)
    arith, geo = simulate_averages(S0, r, T, vol, q, n_fixings, n_paths,
                                   antithetic=antithetic, seed=seed)

    disc = np.exp(-r*T)
    Y = disc * _payoff(arith, K, option_type)
    X = disc * _payoff(geo, K, option_type)

    if antithetic:
        # Same reasoning as mc_price: the iid samples are the pair averages.
        half = len(Y) // 2
        Y = (Y[:half] + Y[half:]) / 2
        X = (X[:half] + X[half:]) / 2

    if control_variate:
        cov = np.cov(Y, X)
        b = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        EX = geometric_asian_price(S0, K, r, T, vol, q, option_type, n_fixings)
        Y = Y - b * (X - EX)

    price = Y.mean()
    stderr = Y.std(ddof=1) / np.sqrt(len(Y))
    return price, stderr
