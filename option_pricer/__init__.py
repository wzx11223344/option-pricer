"""
Multi-Model Options Pricing Engine
===================================
Black-Scholes, Binomial Tree, Monte Carlo, Heston, Merton
with automatic Greeks via complex-step differentiation.
"""

from option_pricer.models import (
    BlackScholes,
    BinomialTree,
    MonteCarlo,
    Heston,
    Merton,
)

from option_pricer.greeks import (
    delta,
    gamma,
    vega,
    theta,
    rho_greek as rho,
    all_greeks,
    verify_greeks,
)

__version__ = "1.0.0"
__all__ = [
    "BlackScholes",
    "BinomialTree",
    "MonteCarlo",
    "Heston",
    "Merton",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "all_greeks",
    "verify_greeks",
]
