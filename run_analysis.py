"""Run a reproducible Black-Scholes vs. Monte Carlo comparison and save a plot."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pricing import black_scholes_price, monte_carlo_payoffs, monte_carlo_price


# Illustrative inputs for a one-year at-the-money European call.
SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
RISK_FREE_RATE = 0.05
VOLATILITY = 0.20
DIVIDEND_YIELD = 0.0


def main() -> None:
    analytical = black_scholes_price(
        SPOT, STRIKE, MATURITY, RISK_FREE_RATE, VOLATILITY,
        option_type="call", dividend_yield=DIVIDEND_YIELD,
    )
    estimate, standard_error = monte_carlo_price(
        SPOT, STRIKE, MATURITY, RISK_FREE_RATE, VOLATILITY,
        option_type="call", n_simulations=250_000, seed=42,
        dividend_yield=DIVIDEND_YIELD,
    )

    print("European call option pricing example")
    print(f"Spot: ${SPOT:.2f} | Strike: ${STRIKE:.2f} | Maturity: {MATURITY:.2f} years")
    print(f"Risk-free rate: {RISK_FREE_RATE:.1%} | Volatility: {VOLATILITY:.1%}")
    print(f"Black-Scholes analytical value : ${analytical:.4f}")
    print(f"Monte Carlo estimate           : ${estimate:.4f}")
    print(f"Monte Carlo standard error     : ${standard_error:.4f}")
    print(f"Absolute difference            : ${abs(estimate - analytical):.4f}")

    # Plot convergence using prefixes from one fixed seeded simulation sample.
    discounted_payoffs = monte_carlo_payoffs(
        SPOT, STRIKE, MATURITY, RISK_FREE_RATE, VOLATILITY,
        option_type="call", n_simulations=250_000, seed=42,
        dividend_yield=DIVIDEND_YIELD,
    )
    counts = np.array([500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000])
    estimates = np.array([np.mean(discounted_payoffs[:n]) for n in counts])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(counts, estimates, marker="o", linewidth=1.6, label="Monte Carlo estimate")
    ax.axhline(analytical, linestyle="--", linewidth=1.6, label=f"Black-Scholes value (${analytical:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths (log scale)")
    ax.set_ylabel("Estimated option value ($)")
    ax.set_title("Monte Carlo convergence toward the Black-Scholes value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "monte_carlo_convergence.png"
    fig.savefig(output_path, dpi=180)
    print(f"\nSaved convergence chart to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
