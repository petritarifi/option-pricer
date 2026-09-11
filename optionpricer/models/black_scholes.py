import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
def d1_d2(S0, K, r, T, vol, q):
    """Compute d1 and d2 for the Black-Scholes-Merton formula."""
    d1 = (np.log(S0/K) + (r - q + 0.5*vol**2)*T)/(vol * np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return (d1, d2)


def bsm_price(S0, K, r, T, vol, q, option_type):
    """
    Black-Scholes-Merton price of a European call or put option.

    Parameters
    ----------
    S0 : float
        Current spot price of the underlying.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity, in years.
    vol : float
        Annualized volatility of the underlying.
    q : float
        Continuous dividend yield.
    option_type : str
        "call" or "put".

    Returns
    -------
    float
        Price of the option.
    """
    d1, d2 = d1_d2(S0, K, r, T, vol, q)
    if option_type == "call":
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K* np.exp(-r*T) * norm.cdf(d2)
    elif option_type == "put":
        return K* np.exp(-r*T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type has to be 'call' or 'put', received : {option_type}")

def black76_price(F0, K, r, T, vol, option_type):
    """
    Black-76 price of a European call or put on a futures or forward price.

    A futures contract costs nothing to enter, so its risk-neutral drift is
    zero: Black-76 is Black-Scholes-Merton with the cost of carry removed,
    i.e. q = r.

    Parameters
    ----------
    F0 : float
        Current futures (or forward) price for delivery at the option maturity.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate, used for discounting only.
    T : float
        Time to maturity, in years.
    vol : float
        Annualized volatility of the futures price.
    option_type : str
        "call" or "put".

    Returns
    -------
    float
        Price of the option.
    """
    d1, d2 = d1_d2(F0, K, r, T, vol, q=r)
    if option_type == "call":
        return np.exp(-r*T) * (F0 * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == "put":
        return np.exp(-r*T) * (K * norm.cdf(-d2) - F0 * norm.cdf(-d1))
    else:
        raise ValueError(f"option_type has to be 'call' or 'put', received : {option_type}")

def bsm_greeks(S0, K, r, T, vol, q, option_type):
    """
    Analytical Black-Scholes-Merton Greeks for a European call or put.

    Returns a dict with:
        delta : dPrice/dS
        gamma : d^2Price/dS^2
        vega  : dPrice/dvol
        theta : dPrice/dT
        rho   : dPrice/dr
    """
    d1, d2 = d1_d2(S0, K, r, T, vol, q)
    pdf_d1 = norm.pdf(d1)

    gamma = (np.exp(-q*T) * pdf_d1) / (S0 * vol * np.sqrt(T))
    vega = S0 * np.exp(-q*T) * pdf_d1 * np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1) * np.exp(-q * T)
        theta = (
            -(S0 * np.exp(-q*T) * pdf_d1 * vol) / (2*np.sqrt(T))
            - r * K * np.exp(-r*T) * norm.cdf(d2)
            + q * S0 * np.exp(-q*T) * norm.cdf(d1)
        )
        rho = K * T * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == "put":
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        theta = (
            -(S0 * np.exp(-q*T) * pdf_d1 * vol) / (2*np.sqrt(T))
            + r * K * np.exp(-r*T) * norm.cdf(-d2)
            - q * S0 * np.exp(-q*T) * norm.cdf(-d1)
        )
        rho = -K * T * np.exp(-r*T) * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type has to be 'call' or 'put', received : {option_type}")

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

def implied_vol(price, S0, K, r, T, q, option_type,
                 vol_lower=1e-6, vol_upper=5.0, tol=1e-8, max_iter=100):
    """
    Implied volatility that reprices a European option to a given market price.

    Solves bsm_price(S0, K, r, T, vol, q, option_type) = price for vol using
    Brent's method.

    Parameters
    ----------
    price : float
        Observed market price of the option.
    S0 : float
        Current spot price of the underlying.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity, in years.
    q : float
        Continuous dividend yield.
    option_type : str
        "call" or "put".
    vol_lower : float, optional
        Lower bound of the volatility search bracket.
    vol_upper : float, optional
        Upper bound of the volatility search bracket.
    tol : float, optional
        Absolute tolerance on the price residual passed to Brent's method.
    max_iter : int, optional
        Maximum number of Brent iterations.

    Returns
    -------
    float
        Implied volatility.

    Raises
    ------
    ValueError
        If option_type is invalid, or if price is outside the no-arbitrage
        bounds implied by S0, K, r, T, q.
    """
    if option_type == "call":
        lower_bound = max(S0 * np.exp(-q*T) - K * np.exp(-r*T), 0.0)
        upper_bound = S0 * np.exp(-q*T)
    elif option_type == "put":
        lower_bound = max(K * np.exp(-r*T) - S0 * np.exp(-q*T), 0.0)
        upper_bound = K * np.exp(-r*T)
    else:
        raise ValueError(f"option_type has to be 'call' or 'put', received: {option_type}")

    if not (lower_bound < price < upper_bound):
        raise ValueError(
            f"price {price} is outside the no-arbitrage bounds "
            f"({lower_bound:.6f}, {upper_bound:.6f}) for the given S0, K, r, T, q"
        )

    def objective(vol):
        return bsm_price(S0, K, r, T, vol, q, option_type) - price

    return brentq(objective, vol_lower, vol_upper, xtol=tol, maxiter=max_iter)