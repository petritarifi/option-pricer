import numpy as np
import pytest

from optionpricer.models.black_scholes import d1_d2

from optionpricer.models.black_scholes import bsm_price

from optionpricer.models.black_scholes import bsm_greeks

from optionpricer.models.black_scholes import implied_vol

from optionpricer.models.black_scholes import black76_price

from optionpricer.models.binomial import binomial_price

def test_d1_d2_known_values():
    d1, d2 = d1_d2(S0=100, K=100, r=0.02, T=1, vol=0.2, q=0.0)
    assert d1 == pytest.approx(0.20, abs=1e-3)
    assert d2 == pytest.approx(0.00, abs=1e-3)

def test_put_call_parity():
    S0, K, r, T, vol, q = 105, 100, 0.03, 0.75, 0.25, 0.01
    call = bsm_price(S0, K, r, T, vol, q, "call")
    put = bsm_price(S0, K, r, T, vol, q, "put")
    lhs = call - put
    rhs = S0 * np.exp(-q*T) - K * np.exp(-r*T)
    assert lhs == pytest.approx(rhs, abs=1e-8)

def test_delta_matches_finite_difference():
    S0, K, r, T, vol, q = 100, 100, 0.02, 1, 0.2, 0.0
    eps = 1e-4
    price_up = bsm_price(S0 + eps, K, r, T, vol, q, "call")
    price_down = bsm_price(S0 - eps, K, r, T, vol, q, "call")
    numeric_delta = (price_up - price_down) / (2 * eps)

    analytic_delta = bsm_greeks(S0, K, r, T, vol, q, "call")["delta"]
    assert numeric_delta == pytest.approx(analytic_delta, abs=1e-3)

def test_implied_vol_round_trip():
    S0, K, r, T, q, option_type = 100, 110, 0.03, 1.0, 0.01, "call"
    true_vol = 0.27
    price = bsm_price(S0, K, r, T, true_vol, q, option_type)
    recovered_vol = implied_vol(price, S0, K, r, T, q, option_type)
    assert recovered_vol == pytest.approx(true_vol, abs=1e-6)

def test_implied_vol_out_of_bounds_raises():
    S0, K, r, T, q, option_type = 100, 100, 0.02, 1.0, 0.0, "call"
    with pytest.raises(ValueError):
        implied_vol(price=200, S0=S0, K=K, r=r, T=T, q=q, option_type=option_type)

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_black76_on_forward_matches_bsm_on_spot(option_type):
    # A European option on a forward expiring at T is the same claim as the
    # option on spot, once F0 = S0 * exp((r - q) T).
    S0, K, r, T, vol, q = 100, 95, 0.04, 0.5, 0.35, 0.02
    F0 = S0 * np.exp((r - q) * T)
    black76 = black76_price(F0, K, r, T, vol, option_type)
    bsm = bsm_price(S0, K, r, T, vol, q, option_type)
    assert black76 == pytest.approx(bsm, abs=1e-10)

def test_black76_put_call_parity():
    F0, K, r, T, vol = 80, 75, 0.04, 0.5, 0.35
    call = black76_price(F0, K, r, T, vol, "call")
    put = black76_price(F0, K, r, T, vol, "put")
    assert call - put == pytest.approx(np.exp(-r*T) * (F0 - K), abs=1e-10)

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_black76_matches_binomial_on_futures(option_type):
    # Independent implementation: a CRR tree with q = r has the zero drift of a futures price.
    F0, K, r, T, vol = 80, 75, 0.04, 0.5, 0.35
    black76 = black76_price(F0, K, r, T, vol, option_type)
    tree = binomial_price(F0, K, r, T, vol, 2000, r, option_type, "european")
    assert tree == pytest.approx(black76, abs=1e-3)

def test_black76_invalid_option_type_raises():
    with pytest.raises(ValueError):
        black76_price(80, 75, 0.04, 0.5, 0.35, "straddle")
