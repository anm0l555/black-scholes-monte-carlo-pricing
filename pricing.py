"""Black-Scholes and Monte Carlo option-pricing utilities.

This is an educational implementation for European options under the
Black-Scholes assumptions. It is not intended for trading or investment use.
"""

from __future__ import annotations

from math import erf, exp, log, pi, sqrt
from typing import Literal

import numpy as np

OptionType = Literal["call", "put"]


def _validate_inputs(
    spot: float, strike: float, maturity: float, rate: float, volatility: float, dividend_yield: float
) -> None:
    if spot <= 0:
        raise ValueError("spot must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if maturity < 0:
        raise ValueError("maturity cannot be negative")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")
    if not all(np.isfinite(x) for x in (spot, strike, maturity, rate, volatility, dividend_yield)):
        raise ValueError("all inputs must be finite numbers")


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Return the Black-Scholes value of a European call or put.

    Parameters use annualized rates and volatility, and maturity is in years.
    ``rate`` and ``dividend_yield`` are continuously compounded.
    """
    _validate_inputs(spot, strike, maturity, rate, volatility, dividend_yield)
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    if maturity == 0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)

    # With zero volatility, the terminal price is deterministic under the model.
    if volatility == 0:
        discounted_spot = spot * exp(-dividend_yield * maturity)
        discounted_strike = strike * exp(-rate * maturity)
        if option_type == "call":
            return max(discounted_spot - discounted_strike, 0.0)
        return max(discounted_strike - discounted_spot, 0.0)

    vol_sqrt_t = volatility * sqrt(maturity)
    d1 = (log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if option_type == "call":
        return spot * exp(-dividend_yield * maturity) * _normal_cdf(d1) - strike * exp(-rate * maturity) * _normal_cdf(d2)
    return strike * exp(-rate * maturity) * _normal_cdf(-d2) - spot * exp(-dividend_yield * maturity) * _normal_cdf(-d1)


def monte_carlo_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType = "call",
    n_simulations: int = 100_000,
    seed: int | None = 42,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Estimate a European option value and its Monte Carlo standard error.

    Simulates the terminal price under risk-neutral geometric Brownian motion.
    Returns ``(price_estimate, standard_error)``. A fixed seed makes the
    illustrative run reproducible.
    """
    _validate_inputs(spot, strike, maturity, rate, volatility, dividend_yield)
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if n_simulations < 2:
        raise ValueError("n_simulations must be at least 2")

    if maturity == 0 or volatility == 0:
        exact = black_scholes_price(spot, strike, maturity, rate, volatility, option_type, dividend_yield)
        return exact, 0.0

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_simulations)
    terminal_spot = spot * np.exp(
        (rate - dividend_yield - 0.5 * volatility**2) * maturity
        + volatility * sqrt(maturity) * z
    )

    if option_type == "call":
        payoffs = np.maximum(terminal_spot - strike, 0.0)
    else:
        payoffs = np.maximum(strike - terminal_spot, 0.0)

    discounted_payoffs = exp(-rate * maturity) * payoffs
    estimate = float(np.mean(discounted_payoffs))
    standard_error = float(np.std(discounted_payoffs, ddof=1) / sqrt(n_simulations))
    return estimate, standard_error


def monte_carlo_payoffs(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: OptionType = "call",
    n_simulations: int = 250_000,
    seed: int | None = 42,
    dividend_yield: float = 0.0,
) -> np.ndarray:
    """Return simulated discounted payoffs, useful for studying convergence."""
    _validate_inputs(spot, strike, maturity, rate, volatility, dividend_yield)
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if n_simulations < 2:
        raise ValueError("n_simulations must be at least 2")

    if maturity == 0 or volatility == 0:
        value = black_scholes_price(spot, strike, maturity, rate, volatility, option_type, dividend_yield)
        return np.full(n_simulations, value, dtype=float)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_simulations)
    terminal_spot = spot * np.exp(
        (rate - dividend_yield - 0.5 * volatility**2) * maturity
        + volatility * sqrt(maturity) * z
    )
    if option_type == "call":
        payoffs = np.maximum(terminal_spot - strike, 0.0)
    else:
        payoffs = np.maximum(strike - terminal_spot, 0.0)
    return exp(-rate * maturity) * payoffs
