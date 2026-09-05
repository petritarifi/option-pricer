import numpy as np
from scipy.stats import norm
def d1_d2(S0, K, r, T, vol):
    """Compute d1 and d2 for the Black-Scholes-Merton formula."""
    d1 = (np.log(S0/K) + (r + 0.5 *vol**2)*T)/(vol * np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return (d1,d2)


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
    d1, d2 = d1_d2(S0, K, r, T, vol)
    if option_type == "call":
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K* np.exp(-r*T) * norm.cdf(d2)
    elif option_type == "put":
        return K* np.exp(-r*T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
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
    d1, d2 = d1_d2(S0, K, r, T, vol)
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