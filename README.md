# Multi-Model Options Pricer

**Black-Scholes, Binomial Tree, Monte Carlo, Heston, Merton** — with automatic Greeks via complex-step differentiation. Pure Python/NumPy/SciPy. No QuantLib dependency.

## Features

- **5 Pricing Models**: Black-Scholes (1973), Binomial Tree CRR (1979), Monte Carlo with variance reduction, Heston stochastic volatility (1993), Merton jump-diffusion (1976)
- **Automatic Greeks**: Complex-step differentiation computes delta, gamma, vega, theta, rho from ANY pricing function — no manual derivative formulas needed
- **Volatility Surface**: Build implied vol surfaces from market data, cubic spline interpolation, smile/skew/term structure analysis
- **Interactive Dashboards**: Plotly-based P&L heatmaps, Greeks surfaces, model comparison, time decay, 3D vol surface
- **CLI Tool**: `pricing_cli.py` with `price`, `greeks`, `iv`, `surface`, `compare`, `dashboard` subcommands
- **Model Calibration**: Least-squares calibration to market prices for Heston and Merton models

## Installation

```bash
cd option-pricer
pip install -r requirements.txt
```

## Quick Start

```bash
# Price an option
python pricing_cli.py price --model bs --S 100 --K 105 --T 0.5 --r 0.03 --sigma 0.2

# Compute Greeks (complex-step, verified against analytic)
python pricing_cli.py greeks --verify --S 100 --K 105 --T 0.5 --r 0.03 --sigma 0.2

# Implied volatility
python pricing_cli.py iv --price 3.5 --S 100 --K 105 --T 0.5 --r 0.03

# Compare models
python pricing_cli.py compare --models bs,heston,merton --S 100 --K 105 --T 0.3

# Run full demo
python examples/demo.py
```

## CLI Reference

### Price Command

```bash
python pricing_cli.py price --model <MODEL> [OPTIONS]

Models:
  bs, blackscholes     Black-Scholes analytic pricing
  tree, binomial       CRR binomial tree (European & American)
  mc, montecarlo       Monte Carlo simulation
  heston               Heston stochastic volatility
  merton               Merton jump-diffusion

Options:
  --S FLOAT            Spot price (default: 100)
  --K FLOAT            Strike price (default: 105)
  --T FLOAT            Time to maturity in years (default: 0.5)
  --r FLOAT            Risk-free rate (default: 0.03)
  --sigma FLOAT        Volatility (required for bs/tree/mc/merton)
  --put                Price a put option (default: call)
  --json               Output as JSON

Model-specific options:
  --N INT              Steps/paths for tree and MC models
  --style {european,american}  Option style for binomial tree
  --no-antithetic      Disable antithetic variates in MC
  --control-variate    Enable control variate in MC
  --v0 FLOAT           Initial variance (Heston)
  --kappa FLOAT        Mean reversion speed (Heston)
  --theta-v FLOAT      Long-run variance (Heston)
  --sigma-v FLOAT      Vol-of-vol (Heston)
  --rho-v FLOAT        Correlation (Heston)
  --lam FLOAT          Jump intensity (Merton)
  --mu-j FLOAT         Mean jump size (Merton)
  --sigma-j FLOAT      Jump size std dev (Merton)
```

### Other Commands

```bash
python pricing_cli.py greeks [--verify] --S 100 --K 105 --T 0.5 --r 0.03 --sigma 0.2
python pricing_cli.py iv --price 3.5 --S 100 --K 105 --T 0.5 --r 0.03
python pricing_cli.py surface --ticker 510050 [--plot]
python pricing_cli.py compare --models bs,heston,merton --S 100 --K 105 --T 0.3
python pricing_cli.py dashboard --S 100 --K 105 --T 0.5 --r 0.03 --sigma 0.2
```

## Python API

```python
from option_pricer.models import BlackScholes, BinomialTree, MonteCarlo, Heston, Merton
from option_pricer.greeks import compute_cs_greeks, verify_greeks
from option_pricer.vol_surface import VolSurface, synthetic_surface
from option_pricer.dashboard import pnl_heatmap, model_comparison, full_dashboard

# Black-Scholes
call = BlackScholes.call_price(S=100, K=105, T=0.5, r=0.03, sigma=0.2)
put = BlackScholes.put_price(S=100, K=105, T=0.5, r=0.03, sigma=0.2)
iv = BlackScholes.implied_vol(call, S=100, K=105, T=0.5, r=0.03)

# Binomial Tree (American put: early exercise premium)
amer_put = BinomialTree.price(100, 105, 0.5, 0.03, 0.2, N=200, call=False, style='american')

# Monte Carlo with confidence intervals
price, se, ci_low, ci_high = MonteCarlo.price(
    100, 105, 0.5, 0.03, 0.2, N=100000, antithetic=True, call=True)

# Heston stochastic volatility
h_price = Heston.price(100, 105, 0.5, 0.03,
                        v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7)

# Merton jump-diffusion
m_price = Merton.price(100, 105, 0.5, 0.03, sigma=0.2,
                        lam=1.0, mu_j=-0.05, sigma_j=0.1)

# Automatic Greeks (complex-step differentiation)
greeks = compute_cs_greeks(100, 105, 0.5, 0.03, 0.2, call=True)
# {'delta': 0.4231, 'gamma': 0.0399, 'vega': 0.1995, 'theta': -4.1067, 'rho': 0.1002}

# Verify against analytic BS Greeks
verification = verify_greeks(100, 105, 0.5, 0.03, 0.2)
# abs_error < 1e-10 for all Greeks
```

## Complex-Step Greeks

The key differentiator: Greeks are computed **automatically** via complex-step differentiation, which avoids the subtractive cancellation that plagues finite differences.

```
f'(x) = Im[f(x + ih)] / h + O(h^2)
f''(x) = 2 * (f(x) - Re[f(x + ih)]) / h^2 + O(h^2)
```

This means you can compute the delta, gamma, vega, theta, and rho of ANY pricing model — including Monte Carlo, Heston, or Merton — by simply passing the pricing function to `greeks.delta()`, `greeks.gamma()`, etc. No manual derivative formulas required.

## Project Structure

```
option-pricer/
├── README.md
├── requirements.txt
├── pricing_cli.py              # CLI entry point
├── option_pricer/
│   ├── __init__.py
│   ├── models.py               # 5 pricing models
│   ├── greeks.py               # Complex-step differentiation
│   ├── vol_surface.py          # Volatility surface
│   └── dashboard.py            # Plotly dashboards
├── examples/
│   └── demo.py                 # Comprehensive demo
└── output/                     # Generated charts
```

## Requirements

- numpy >= 1.21.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- plotly >= 5.0.0 (optional, for dashboards)
- akshare >= 1.10.0 (optional, for market data)
- pyyaml >= 6.0

## License

MIT
