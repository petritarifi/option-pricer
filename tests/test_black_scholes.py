import numpy as np
import pytest

from optionpricer.models.black_scholes import d1_d2

from optionpricer.models.black_scholes import bsm_price

from optionpricer.models.black_scholes import bsm_greeks

def test_d1_d2_known_values():
    d1, d2 = d1_d2(S0=100, K=100, r=0.02, T=1, vol=0.2)
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