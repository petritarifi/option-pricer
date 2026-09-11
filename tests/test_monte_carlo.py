import pytest

from optionpricer.models.black_scholes import bsm_price
from optionpricer.models.monte_carlo import mc_price


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_mc_price_converges_to_bsm(option_type):
    S0, K, r, T, vol, q = 100, 105, 0.03, 0.75, 0.25, 0.01
    bsm = bsm_price(S0, K, r, T, vol, q, option_type)
    mc, se = mc_price(S0, K, r, T, vol, q, option_type, n_paths=200_000, seed=42)

    z = abs(mc - bsm) / se
    assert z < 3.0


def test_mc_stderr_shrinks_with_more_paths():
    S0, K, r, T, vol, q = 100, 100, 0.02, 1.0, 0.2, 0.0
    _, se_small = mc_price(S0, K, r, T, vol, q, "call", n_paths=1_000, seed=1)
    _, se_large = mc_price(S0, K, r, T, vol, q, "call", n_paths=100_000, seed=1)
    assert se_large < se_small


def test_mc_antithetic_reduces_variance():
    S0, K, r, T, vol, q = 100, 100, 0.02, 1.0, 0.2, 0.0
    _, se_plain = mc_price(S0, K, r, T, vol, q, "call", n_paths=20_000,
                            antithetic=False, seed=7)
    _, se_anti = mc_price(S0, K, r, T, vol, q, "call", n_paths=20_000,
                           antithetic=True, seed=7)
    assert se_anti < se_plain


def test_mc_invalid_option_type_raises():
    with pytest.raises(ValueError):
        mc_price(100, 100, 0.02, 1.0, 0.2, 0.0, "straddle", n_paths=1_000)
