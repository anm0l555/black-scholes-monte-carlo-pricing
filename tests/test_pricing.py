import pytest

from pricing import black_scholes_price, monte_carlo_price


PARAMS = dict(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.20)


def test_black_scholes_call_known_example():
    assert black_scholes_price(**PARAMS, option_type="call") == pytest.approx(10.4506, abs=1e-3)


def test_put_call_parity_without_dividends():
    call = black_scholes_price(**PARAMS, option_type="call")
    put = black_scholes_price(**PARAMS, option_type="put")
    expected_difference = PARAMS["spot"] - PARAMS["strike"] * __import__("math").exp(-PARAMS["rate"] * PARAMS["maturity"])
    assert call - put == pytest.approx(expected_difference, abs=1e-8)


def test_monte_carlo_is_consistent_with_analytical_value():
    analytical = black_scholes_price(**PARAMS, option_type="call")
    estimate, standard_error = monte_carlo_price(**PARAMS, option_type="call", n_simulations=200_000, seed=7)
    assert abs(estimate - analytical) < 4.5 * standard_error + 1e-8


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        black_scholes_price(spot=-1, strike=100, maturity=1, rate=0.05, volatility=0.2)
