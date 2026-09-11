import numpy as np
import pytest

from optionpricer.models.black_scholes import bsm_price
from optionpricer.models.asian import (
    geometric_asian_price,
    levy_asian_price,
    mc_asian_price,
    simulate_averages,
)

# Weekly fixings over one year
S0, K, r, T, vol, q, n_fixings = 100, 100, 0.03, 1.0, 0.25, 0.01, 52


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_single_fixing_reduces_to_european(option_type):
    # With one fixing at T the average is S_T: both formulas must collapse to BSM.
    bsm = bsm_price(S0, 105, r, T, vol, q, option_type)
    geo = geometric_asian_price(S0, 105, r, T, vol, q, option_type, n_fixings=1)
    levy = levy_asian_price(S0, 105, r, T, vol, q, option_type, n_fixings=1)
    assert geo == pytest.approx(bsm, abs=1e-10)
    assert levy == pytest.approx(bsm, abs=1e-10)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_geometric_closed_form_matches_mc(option_type):
    # The closed form is E[X] in the control variate, so check it independently.
    _, geo_avg = simulate_averages(S0, r, T, vol, q, n_fixings, n_paths=200_000,
                                   antithetic=False, seed=42)
    payoff = np.maximum(geo_avg - K, 0.0) if option_type == "call" else np.maximum(K - geo_avg, 0.0)
    disc_payoff = np.exp(-r*T) * payoff
    mc = disc_payoff.mean()
    se = disc_payoff.std(ddof=1) / np.sqrt(len(disc_payoff))

    closed_form = geometric_asian_price(S0, K, r, T, vol, q, option_type, n_fixings)
    assert abs(mc - closed_form) / se < 3.0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_control_variate_does_not_bias(option_type):
    # Plain MC and CV estimator on independent paths must agree within noise.
    plain, se_plain = mc_asian_price(S0, K, r, T, vol, q, option_type, n_fixings,
                                     n_paths=200_000, antithetic=False,
                                     control_variate=False, seed=1)
    cv, se_cv = mc_asian_price(S0, K, r, T, vol, q, option_type, n_fixings,
                               n_paths=200_000, seed=2)
    assert abs(plain - cv) / np.hypot(se_plain, se_cv) < 3.0


def test_control_variate_reduces_variance_beyond_antithetic():
    _, se_anti = mc_asian_price(S0, K, r, T, vol, q, "call", n_fixings,
                                n_paths=20_000, control_variate=False, seed=7)
    _, se_cv = mc_asian_price(S0, K, r, T, vol, q, "call", n_fixings,
                              n_paths=20_000, control_variate=True, seed=7)
    # Measured ratio is ~20x in stderr (~500x in variance); 10x leaves margin.
    assert se_cv < se_anti / 10


def test_mc_matches_levy_approximation_atm():
    # Levy is an approximation, not an exact price: its bias is systematic, so
    # the check is a relative tolerance, not a z-score. Measured gap ~0.3% here.
    mc, _ = mc_asian_price(S0, K, r, T, 0.2, q, "call", n_fixings, n_paths=200_000, seed=3)
    levy = levy_asian_price(S0, K, r, T, 0.2, q, "call", n_fixings)
    assert levy == pytest.approx(mc, rel=0.01)


def test_arithmetic_average_dominates_geometric():
    # AM >= GM pathwise, so arithmetic call >= geometric call and the reverse for puts.
    call, se_c = mc_asian_price(S0, K, r, T, vol, q, "call", n_fixings, n_paths=50_000, seed=5)
    put, se_p = mc_asian_price(S0, K, r, T, vol, q, "put", n_fixings, n_paths=50_000, seed=5)
    assert call - 3*se_c > geometric_asian_price(S0, K, r, T, vol, q, "call", n_fixings)
    assert put + 3*se_p < geometric_asian_price(S0, K, r, T, vol, q, "put", n_fixings)


@pytest.mark.parametrize("pricer", [geometric_asian_price, levy_asian_price])
def test_closed_forms_invalid_option_type_raises(pricer):
    with pytest.raises(ValueError):
        pricer(S0, K, r, T, vol, q, "straddle", n_fixings)


def test_mc_asian_invalid_option_type_raises():
    with pytest.raises(ValueError):
        mc_asian_price(S0, K, r, T, vol, q, "straddle", n_fixings, n_paths=1_000)
