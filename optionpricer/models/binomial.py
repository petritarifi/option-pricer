import numpy as np

def binomial_price(S0, K, r, T, vol, n_steps, q, option_type, exercise):
    dt = T / n_steps
    u = np.exp(vol*np.sqrt(dt))
    d = 1/u
    p = (np.exp((r-q)*dt)-d)/(u-d)

    if not (0 < p < 1):
        raise ValueError(f"Invalid risk-neutral probability p={p}, check your inputs.")

    disc = np.exp(-r*dt)

    # Terminal asset prices (n_steps+1 nodes, j = number of up moves)
    j = np.arange(n_steps + 1)
    ST = S0 * u**j * d**(n_steps - j)

    # Terminal payoffs
    if option_type == "call":
        values = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        values = np.maximum(K - ST, 0.0)
    else:
        raise ValueError(f"Unknown option_type: {option_type}")

    # Backward induction through the tree
    for i in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])

        if exercise == "american":
            j = np.arange(i + 1)
            S_i = S0 * u**j * d**(i - j)
            if option_type == "call":
                intrinsic = np.maximum(S_i - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_i, 0.0)
            values = np.maximum(values, intrinsic)
        elif exercise != "european":
            raise ValueError(f"Unknown exercise: {exercise}")

    return values[0]