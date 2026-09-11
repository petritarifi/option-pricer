"""
Plot the convergence of the Monte Carlo option price towards the
Black-Scholes analytical price as the number of simulated paths grows.

Usage (from the project root):
    python -m optionpricer.scripts.plot_mc_convergence
"""
import numpy as np
import matplotlib.pyplot as plt

from optionpricer.models.black_scholes import bsm_price
from optionpricer.models.monte_carlo import mc_price

def convergence_data(S0, K, r, T, vol, q, option_type, n_paths_list, seed=42):
    bsm = bsm_price(S0, K, r, T, vol, q, option_type)

    prices, ci_low, ci_high = [], [], []
    for n in n_paths_list:
        price, se = mc_price(S0, K, r, T, vol, q, option_type, n_paths=n, seed=seed)
        prices.append(price)
        ci_low.append(price - 1.96 * se)
        ci_high.append(price + 1.96 * se)

    return bsm, np.array(prices), np.array(ci_low), np.array(ci_high)


def plot_convergence(S0=100, K=105, r=0.03, T=0.75, vol=0.25, q=0.015,
                      option_type="call", save_path="mc_convergence.png"):
    n_paths_list = [10**i for i in range(2, 7)]  # 100 -> 1,000,000

    bsm, prices, ci_low, ci_high = convergence_data(
        S0, K, r, T, vol, q, option_type, n_paths_list
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(n_paths_list, prices, marker="o", color="#1f3864", label="Monte Carlo price")
    ax.fill_between(n_paths_list, ci_low, ci_high, color="#1f3864", alpha=0.15,
                     label="95% CI")
    ax.axhline(bsm, color="crimson", linestyle="--", label="Black-Scholes price")

    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths")
    ax.set_ylabel("Option price")
    ax.set_title(f"Monte Carlo convergence ({option_type}, S0={S0}, K={K}, T={T}y)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    return fig


if __name__ == "__main__":
    plot_convergence()