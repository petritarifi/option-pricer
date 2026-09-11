import numpy as np
def mc_price(S0, K, r, T, vol, q, option_type, n_paths, antithetic=True, seed=None):
    """
    European option price by Monte Carlo simulation of terminal GBM prices.
    Returns (price, stderr) — use ~1.96*stderr for a 95% CI.
    """
    rng = np.random.default_rng(seed)

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal(half)
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5*vol**2) * T
    diffusion = vol*np.sqrt(T) * Z
    ST = S0 * np.exp(drift + diffusion)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError(f"option_type has to be 'call' or 'put', received: {option_type}")

    disc_payoff = np.exp(-r*T) * payoff
    price = disc_payoff.mean()

    if antithetic:
        # Antithetic pairs (Z, -Z) are negatively correlated, not independent,
        # so the standard error must come from the paired averages, not from
        # treating all 2*half samples as iid.
        half = len(disc_payoff) // 2
        pair_means = (disc_payoff[:half] + disc_payoff[half:]) / 2
        stderr = pair_means.std(ddof=1) / np.sqrt(half)
    else:
        stderr = disc_payoff.std(ddof=1) / np.sqrt(len(disc_payoff))

    return price, stderr