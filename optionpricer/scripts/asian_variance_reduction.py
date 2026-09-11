"""
Measure the variance reduction of the arithmetic Asian Monte Carlo pricer
(plain, antithetic, geometric control variate, both), check over many seeds
that the control variate introduces no bias, and compare the price against
the geometric closed form and the Levy approximation.

Usage (from the project root):
    python -m optionpricer.scripts.asian_variance_reduction
"""
import numpy as np
import matplotlib.pyplot as plt

from optionpricer.models.asian import (
    geometric_asian_price,
    levy_asian_price,
    mc_asian_price,
    simulate_averages,
)

ESTIMATORS = {
    "Plain MC": dict(antithetic=False, control_variate=False),
    "Antithetic": dict(antithetic=True, control_variate=False),
    "Control variate": dict(antithetic=False, control_variate=True),
    "Antithetic + control variate": dict(antithetic=True, control_variate=True),
}

# Only three series are plotted: CV alone and antithetic + CV are
# indistinguishable on a log scale, so the table carries that comparison.
PLOTTED = {
    "Plain MC": "#2a78d6",
    "Antithetic": "#eb6834",
    "Antithetic + control variate": "#1baf7a",
}


def variance_reduction_table(S0, K, r, T, vol, q, option_type, n_fixings,
                             n_paths=100_000, seed=42):
    rows = []
    for name, flags in ESTIMATORS.items():
        price, se = mc_asian_price(S0, K, r, T, vol, q, option_type, n_fixings,
                                   n_paths=n_paths, seed=seed, **flags)
        rows.append((name, price, se))

    se_plain = rows[0][2]
    print(f"\nArithmetic Asian {option_type}, S0={S0}, K={K}, T={T}y, vol={vol}, "
          f"{n_fixings} fixings, {n_paths:,} paths")
    print(f"{'Estimator':<30}{'Price':>10}{'Stderr':>12}{'Variance reduction':>20}")
    for name, price, se in rows:
        print(f"{name:<30}{price:>10.4f}{se:>12.6f}{(se_plain / se)**2:>19.0f}x")

    geo = geometric_asian_price(S0, K, r, T, vol, q, option_type, n_fixings)
    levy = levy_asian_price(S0, K, r, T, vol, q, option_type, n_fixings)
    best_price, best_se = rows[-1][1], rows[-1][2]
    print(f"\n{'Geometric Asian (closed form)':<30}{geo:>10.4f}")
    print(f"{'Levy approximation':<30}{levy:>10.4f}   "
          f"gap vs MC: {levy - best_price:+.4f} ({(levy - best_price) / best_se:+.0f} stderr)")
    return rows


def bias_check(S0, K, r, T, vol, q, option_type, n_fixings, n_seeds=40, n_paths=100_000):
    """
    Two separate unbiasedness checks, each repeated over n_seeds independent seeds:
    (1) closed-form geometric price vs plain MC of the geometric-average option;
    (2) control-variate estimator vs plain MC of the arithmetic-average option.
    If both are unbiased with calibrated stderrs, the z-scores are ~ N(0, 1).
    """
    geo = geometric_asian_price(S0, K, r, T, vol, q, option_type, n_fixings)
    z_geo, z_cv = [], []
    for s in range(n_seeds):
        _, geo_avg = simulate_averages(S0, r, T, vol, q, n_fixings, n_paths,
                                       antithetic=False, seed=1000 + s)
        payoff = np.maximum(geo_avg - K, 0.0) if option_type == "call" else np.maximum(K - geo_avg, 0.0)
        X = np.exp(-r*T) * payoff
        z_geo.append((X.mean() - geo) / (X.std(ddof=1) / np.sqrt(n_paths)))

        plain, se_plain = mc_asian_price(S0, K, r, T, vol, q, option_type, n_fixings, n_paths,
                                         antithetic=False, control_variate=False, seed=2000 + s)
        cv, se_cv = mc_asian_price(S0, K, r, T, vol, q, option_type, n_fixings, n_paths,
                                   seed=3000 + s)
        z_cv.append((cv - plain) / np.hypot(se_plain, se_cv))

    print(f"\nUnbiasedness over {n_seeds} independent seeds ({n_paths:,} paths each), "
          f"z-scores should be ~ N(0, 1):")
    for name, z in [("(1) geometric closed form vs plain MC", z_geo),
                    ("(2) control variate vs plain MC", z_cv)]:
        z = np.array(z)
        print(f"{name:<42} mean {z.mean():+.2f} (± {1/np.sqrt(n_seeds):.2f})   std {z.std(ddof=1):.2f}")


def plot_stderr(S0, K, r, T, vol, q, option_type, n_fixings, seed=42,
                save_path="asian_variance_reduction.png"):
    n_paths_list = np.array([10**i for i in range(3, 7)])  # 1,000 -> 1,000,000

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color in PLOTTED.items():
        ses = [mc_asian_price(S0, K, r, T, vol, q, option_type, n_fixings,
                              n_paths=n, seed=seed, **ESTIMATORS[name])[1]
               for n in n_paths_list]
        ax.plot(n_paths_list, ses, marker="o", markersize=5, linewidth=2,
                color=color, label=name)
        ax.annotate(name, (n_paths_list[-1], ses[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color="#52514e")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(right=n_paths_list[-1] * 12)  # room for the direct labels
    ax.set_xlabel("Number of simulated paths")
    ax.set_ylabel("Standard error of the price estimate")
    ax.set_title(f"Arithmetic Asian {option_type}: standard error by estimator "
                 f"(K={K}, vol={vol}, {n_fixings} fixings)")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(alpha=0.3, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
    return fig


if __name__ == "__main__":
    # Weekly fixings over one year, at-the-money
    params = dict(S0=100, K=100, r=0.03, T=1.0, vol=0.25, q=0.0,
                  option_type="call", n_fixings=52)
    variance_reduction_table(**params)
    bias_check(**params)
    plot_stderr(**params)
