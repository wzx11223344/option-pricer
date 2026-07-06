#!/usr/bin/env python
"""
Multi-Model Options Pricing CLI
================================
Command-line interface for pricing options across 5 models,
computing automatic Greeks, implied volatility, volatility surface,
and model comparison.

Usage:
    python pricing_cli.py price --model bs --S 100 --K 105 --T 0.5 --r 0.03 --sigma 0.2
    python pricing_cli.py price --model heston --S 100 --K 105 --T 0.5 --r 0.03
    python pricing_cli.py greeks --model bs --S 100 --K 105 --T 0.5 --r 0.03 --sigma 0.2
    python pricing_cli.py iv --price 3.5 --S 100 --K 105 --T 0.5 --r 0.03
    python pricing_cli.py surface --ticker 510050
    python pricing_cli.py compare --models bs,heston,merton --S 100 --K 105 --T 0.3
    python pricing_cli.py dashboard --S 100 --K 105 --T 0.5 --r 0.03 --sigma 0.2
"""

import argparse
import json
import sys
import os

import numpy as np

from option_pricer.models import (
    BlackScholes,
    BinomialTree,
    MonteCarlo,
    Heston,
    Merton,
)
from option_pricer.greeks import (
    compute_cs_greeks,
    verify_greeks,
    all_greeks,
)


# ---------------------------------------------------------------------------
# Price command
# ---------------------------------------------------------------------------

def cmd_price(args):
    """Price an option using the specified model."""
    S = args.S
    K = args.K
    T = args.T
    r = args.r
    sigma = args.sigma
    q = getattr(args, 'q', 0.0)
    call = not args.put

    model = args.model.lower()
    opt_type = 'Call' if call else 'Put'

    results = {
        'model': model,
        'type': opt_type,
        'parameters': {'S': S, 'K': K, 'T': T, 'r': r},
    }

    if model == 'bs' or model == 'blackscholes':
        if sigma is None:
            print("ERROR: --sigma is required for Black-Scholes model")
            sys.exit(1)
        price = (BlackScholes.call_price(S, K, T, r, sigma, q) if call else
                 BlackScholes.put_price(S, K, T, r, sigma, q))
        results['sigma'] = sigma
        results['price'] = float(price)

    elif model == 'tree' or model == 'binomial':
        if sigma is None:
            print("ERROR: --sigma is required for Binomial Tree model")
            sys.exit(1)
        N = getattr(args, 'N', 200)
        style = getattr(args, 'style', 'european')
        price = BinomialTree.price(S, K, T, r, sigma, N=N,
                                   call=call, style=style, q=q)
        results['sigma'] = sigma
        results['N'] = N
        results['style'] = style
        results['price'] = float(price)

    elif model == 'mc' or model == 'montecarlo':
        if sigma is None:
            print("ERROR: --sigma is required for Monte Carlo model")
            sys.exit(1)
        N = getattr(args, 'N', 100000)
        antithetic = not args.no_antithetic
        price, std_err, ci_low, ci_high = MonteCarlo.price(
            S, K, T, r, sigma, N=N, antithetic=antithetic,
            call=call, control_variate=args.control_variate, q=q)
        results['sigma'] = sigma
        results['N'] = N
        results['antithetic'] = antithetic
        results['price'] = float(price)
        results['std_error'] = float(std_err)
        results['ci_95'] = [float(ci_low), float(ci_high)]

    elif model == 'heston':
        v0 = getattr(args, 'v0', sigma**2 if sigma else 0.04)
        kappa = getattr(args, 'kappa', 2.0)
        theta_v = getattr(args, 'theta_v', sigma**2 if sigma else 0.04)
        sigma_v = getattr(args, 'sigma_v', 0.3)
        rho_v = getattr(args, 'rho_v', -0.7)
        price = Heston.price(S, K, T, r, v0, kappa, theta_v, sigma_v,
                             rho_v, call=call, q=q)
        results['heston_params'] = {
            'v0': v0, 'kappa': kappa, 'theta': theta_v,
            'sigma_v': sigma_v, 'rho': rho_v,
        }
        results['price'] = float(price)

    elif model == 'merton':
        if sigma is None:
            print("ERROR: --sigma is required for Merton model")
            sys.exit(1)
        lam = getattr(args, 'lam', 1.0)
        mu_j = getattr(args, 'mu_j', -0.05)
        sigma_j = getattr(args, 'sigma_j', 0.1)
        price = Merton.price(S, K, T, r, sigma, lam, mu_j, sigma_j,
                             call=call, q=q)
        results['sigma'] = sigma
        results['merton_params'] = {
            'lam': lam, 'mu_j': mu_j, 'sigma_j': sigma_j
        }
        results['price'] = float(price)

    else:
        print(f"ERROR: Unknown model '{model}'. "
              f"Choose: bs, tree, mc, heston, merton")
        sys.exit(1)

    # Print results
    if args.json:
        print(json.dumps(results, indent=2, default=float))
    else:
        print(f"\n{'='*55}")
        print(f"  Option Pricing Result")
        print(f"{'='*55}")
        print(f"  Model:       {results['model'].upper()}")
        print(f"  Type:        {results['type']}")
        print(f"  Spot (S):    {S:.2f}")
        print(f"  Strike (K):  {K:.2f}")
        print(f"  Maturity:    {T:.4f} years")
        print(f"  Rate:        {r:.4f}")
        if 'sigma' in results:
            print(f"  Volatility:  {results['sigma']:.4f}")
        print(f"  {results['type']} Price: {results['price']:.6f}")
        if 'std_error' in results:
            print(f"  Std Error:   {results['std_error']:.6f}")
            print(f"  95% CI:      [{results['ci_95'][0]:.6f}, {results['ci_95'][1]:.6f}]")
        if 'heston_params' in results:
            hp = results['heston_params']
            print(f"  Heston:      v0={hp['v0']:.4f}, kappa={hp['kappa']:.2f}, "
                  f"theta={hp['theta']:.4f}, sigma_v={hp['sigma_v']:.3f}, rho={hp['rho']:.2f}")
        if 'merton_params' in results:
            mp = results['merton_params']
            print(f"  Merton:      lam={mp['lam']:.2f}, mu_j={mp['mu_j']:.3f}, "
                  f"sigma_j={mp['sigma_j']:.3f}")
        print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Greeks command
# ---------------------------------------------------------------------------

def cmd_greeks(args):
    """Compute Greeks for an option."""
    S = args.S
    K = args.K
    T = args.T
    r = args.r
    sigma = args.sigma
    q = getattr(args, 'q', 0.0)
    call = not args.put

    if sigma is None:
        print("ERROR: --sigma is required")
        sys.exit(1)

    opt_type = 'Call' if call else 'Put'

    if args.verify:
        # Verify complex-step against analytic BS Greeks
        result = verify_greeks(S, K, T, r, sigma, q)
        if args.json:
            print(json.dumps(result, indent=2, default=float))
        else:
            print(f"\n{'='*70}")
            print(f"  Complex-Step Greeks Verification ({opt_type})")
            print(f"{'='*70}")
            print(f"  {'Greek':<10} {'Analytic':>14} {'Complex-Step':>14} "
                  f"{'Abs Error':>14} {'Rel Error':>14}")
            print(f"  {'-'*66}")
            for key in ['delta', 'gamma', 'vega', 'theta', 'rho']:
                a = result['analytic'][key]
                c = result['complex_step'][key]
                ae = result['abs_error'][key]
                re = result['rel_error'][key]
                print(f"  {key:<10} {a:14.8f} {c:14.8f} {ae:14.2e} {re:14.2e}")
            print(f"{'='*70}\n")
    else:
        # Compute Greeks using complex-step
        greeks = compute_cs_greeks(S, K, T, r, sigma, q, call)
        price = (BlackScholes.call_price(S, K, T, r, sigma, q) if call else
                 BlackScholes.put_price(S, K, T, r, sigma, q))

        if args.json:
            output = {'price': float(price), 'type': opt_type}
            output.update({k: float(v) for k, v in greeks.items()})
            print(json.dumps(output, indent=2))
        else:
            print(f"\n{'='*55}")
            print(f"  Greeks via Complex-Step Differentiation ({opt_type})")
            print(f"{'='*55}")
            print(f"  S={S:.2f}, K={K:.2f}, T={T:.4f}, r={r:.4f}, sigma={sigma:.4f}")
            print(f"  Price:  {price:.6f}")
            print(f"  {'-'*51}")
            print(f"  Delta:  {greeks['delta']:.6f}")
            print(f"  Gamma:  {greeks['gamma']:.6f}")
            print(f"  Vega:   {greeks['vega']:.6f}")
            print(f"  Theta:  {greeks['theta']:.6f}  (per year)")
            print(f"  Rho:    {greeks['rho']:.6f}")
            print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Implied Volatility command
# ---------------------------------------------------------------------------

def cmd_iv(args):
    """Compute implied volatility from an option price."""
    price = args.price
    S = args.S
    K = args.K
    T = args.T
    r = args.r
    q = getattr(args, 'q', 0.0)

    # Try call first, then put
    iv_call = BlackScholes.implied_vol(price, S, K, T, r, q, call=True)
    iv_put = BlackScholes.implied_vol(price, S, K, T, r, q, call=False)

    if args.json:
        print(json.dumps({
            'price': price,
            'iv_call': float(iv_call) if not np.isnan(iv_call) else None,
            'iv_put': float(iv_put) if not np.isnan(iv_put) else None,
        }, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"  Implied Volatility")
        print(f"{'='*50}")
        print(f"  Market Price: {price:.4f}")
        print(f"  S={S:.2f}, K={K:.2f}, T={T:.4f}, r={r:.4f}")
        if not np.isnan(iv_call):
            print(f"  IV (Call):    {iv_call:.4f} ({iv_call*100:.2f}%)")
        else:
            print(f"  IV (Call):    -- (no solution)")
        if not np.isnan(iv_put):
            print(f"  IV (Put):     {iv_put:.4f} ({iv_put*100:.2f}%)")
        else:
            print(f"  IV (Put):     -- (no solution)")
        print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Vol Surface command
# ---------------------------------------------------------------------------

def cmd_surface(args):
    """Build and plot volatility surface."""
    ticker = args.ticker
    S = args.S
    r = args.r

    print(f"Building volatility surface for {ticker}...")

    try:
        # Try to fetch real option data via akshare
        import akshare as ak
        import pandas as pd

        # Map ticker to SSE/SZSE
        if ticker.startswith('6'):
            symbol = f"sh{ticker}"
        elif ticker.startswith('0') or ticker.startswith('3'):
            symbol = f"sz{ticker}"
        else:
            symbol = ticker

        # Fetch option chain
        try:
            df = ak.option_finance_board(symbol=symbol)
        except Exception:
            # Try alternative: option_sse_codes_sina for SSE
            try:
                df = ak.option_sina_sse_list(
                    symbol=f"合约标的: {ticker}",
                    trade_date=pd.Timestamp.now().strftime('%Y%m')
                )
            except Exception:
                print("WARNING: Could not fetch market data. "
                      "Using synthetic surface for demonstration.")
                from option_pricer.vol_surface import synthetic_surface
                surf = synthetic_surface(S if S else 100.0, r if r else 0.03)
                _plot_and_analyze_surface(surf, args, ticker)
                return

        # Parse option chain data
        print(f"  Fetched {len(df)} option records")

        strikes = []
        maturities = []
        prices_list = []
        types_list = []

        for _, row in df.iterrows():
            try:
                K_val = float(row.get('行权价', row.get('strike', 0)))
                price_val = float(row.get('最新价', row.get('last_price', 0)))
                opt_type = str(row.get('期权类型', row.get('type', 'C')))

                # Parse maturity
                expiry_str = str(row.get('到期日', row.get('expiry_date', '')))
                if expiry_str:
                    expiry = pd.Timestamp(expiry_str)
                    T_val = max((expiry - pd.Timestamp.now()).days / 365.0, 1/365)
                else:
                    T_val = 0.25  # default

                if K_val > 0 and price_val > 0:
                    strikes.append(K_val)
                    maturities.append(T_val)
                    prices_list.append(price_val)
                    types_list.append('call' if opt_type in ('C', 'c', '看涨') else 'put')
            except (ValueError, KeyError):
                continue

        if len(strikes) < 5:
            print("  Too few valid option records, using synthetic surface.")
            from option_pricer.vol_surface import synthetic_surface
            surf = synthetic_surface(S if S else 100.0, r if r else 0.03)
        else:
            from option_pricer.vol_surface import VolSurface
            surf = VolSurface.from_market_prices(
                prices_list, strikes, maturities, types_list,
                S if S else np.median(strikes),
                r if r else 0.03)

        _plot_and_analyze_surface(surf, args, ticker)

    except ImportError:
        print("NOTE: akshare not installed. Using synthetic surface.")
        from option_pricer.vol_surface import synthetic_surface
        surf = synthetic_surface(S if S else 100.0, r if r else 0.03)
        _plot_and_analyze_surface(surf, args, ticker)
    except Exception as e:
        print(f"ERROR: {e}")
        print("Falling back to synthetic surface.")
        from option_pricer.vol_surface import synthetic_surface
        surf = synthetic_surface(S if S else 100.0, r if r else 0.03)
        _plot_and_analyze_surface(surf, args, ticker)


def _plot_and_analyze_surface(surf, args, ticker):
    """Helper: analyze and optionally plot the vol surface."""
    # Smile analysis
    smile = surf.smile_analysis()
    print(f"\n  Volatility Smile/Skew Analysis:")
    print(f"  {'T':>8}  {'ATM IV':>10}  {'Skew':>10}  {'Curvature':>10}  {'Pattern'}")
    print(f"  {'-'*60}")
    for T_val, info in sorted(smile.items()):
        print(f"  {T_val:8.4f}  {info['atm_iv']:10.4f}  {info['skew']:10.4f}  "
              f"{info['curvature']:10.4f}  {info['pattern']}")

    # Term structure
    Ts, atm_ivs = surf.term_structure()
    print(f"\n  Term Structure (ATM IV):")
    for t, iv in zip(Ts, atm_ivs):
        print(f"    T={t:.4f}: {iv:.4f} ({iv*100:.2f}%)")

    # Plot if requested
    if args.plot:
        output_dir = args.output_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'output')
        os.makedirs(output_dir, exist_ok=True)

        try:
            fig_surf = surf.plot_surface(
                title=f"Implied Volatility Surface - {ticker}")
            path_surf = os.path.join(output_dir, f'{ticker}_vol_surface.html')
            fig_surf.write_html(path_surf)
            print(f"\n  Surface plot saved: {path_surf}")

            fig_smile = surf.plot_smile()
            path_smile = os.path.join(output_dir, f'{ticker}_vol_smile.html')
            fig_smile.write_html(path_smile)
            print(f"  Smile plot saved: {path_smile}")

            fig_term = surf.plot_term_structure()
            path_term = os.path.join(output_dir, f'{ticker}_term_structure.html')
            fig_term.write_html(path_term)
            print(f"  Term structure plot saved: {path_term}")
        except ImportError:
            print("\n  NOTE: Install plotly for interactive charts: pip install plotly")


# ---------------------------------------------------------------------------
# Compare command
# ---------------------------------------------------------------------------

def cmd_compare(args):
    """Compare option prices across multiple models."""
    S = args.S
    K = args.K
    T = args.T
    r = args.r
    sigma = args.sigma or 0.2
    q = getattr(args, 'q', 0.0)
    call = not args.put
    models = [m.strip().lower() for m in args.models.split(',')]

    opt_type = 'Call' if call else 'Put'
    results = []

    print(f"\n{'='*70}")
    print(f"  Model Comparison: {opt_type} Option")
    print(f"  S={S:.2f}, K={K:.2f}, T={T:.4f}, r={r:.4f}, sigma={sigma:.4f}")
    print(f"{'='*70}")
    print(f"  {'Model':<20} {'Price':>12} {'Notes'}")
    print(f"  {'-'*65}")

    for model in models:
        try:
            if model in ('bs', 'blackscholes'):
                price = (BlackScholes.call_price(S, K, T, r, sigma, q) if call else
                         BlackScholes.put_price(S, K, T, r, sigma, q))
                note = f'sigma={sigma:.2f}'
            elif model in ('tree', 'binomial'):
                price = BinomialTree.price(S, K, T, r, sigma, N=200,
                                           call=call, q=q)
                note = 'N=200, CRR'
            elif model in ('mc', 'montecarlo'):
                price, se, _, _ = MonteCarlo.price(
                    S, K, T, r, sigma, N=50000, antithetic=True,
                    call=call, q=q, control_variate=False)
                note = f'N=50000, antithetic, SE={se:.4f}'
            elif model == 'heston':
                price = Heston.price(S, K, T, r, v0=sigma**2,
                                     kappa=2.0, theta=sigma**2,
                                     sigma_v=0.3, rho=-0.7,
                                     call=call, q=q)
                note = 'v0=sigma^2, kappa=2, rho=-0.7'
            elif model == 'merton':
                price = Merton.price(S, K, T, r, sigma*0.8,
                                     lam=1.0, mu_j=-0.05,
                                     sigma_j=0.1, call=call, q=q)
                note = 'lam=1, mu_j=-0.05'
            else:
                print(f"  {model:<20} {'SKIP':>12}  Unknown model")
                continue

            results.append({'model': model, 'price': float(price), 'notes': note})
            print(f"  {model:<20} {price:12.6f}  {note}")
        except Exception as e:
            print(f"  {model:<20} {'ERROR':>12}  {str(e)[:35]}")

    # Highlight differences
    if len(results) >= 2:
        prices = [r['price'] for r in results]
        p_min, p_max = min(prices), max(prices)
        spread = p_max - p_min
        print(f"  {'-'*65}")
        print(f"  Model Spread: {spread:.6f} "
              f"(min={p_min:.4f}, max={p_max:.4f})")

    print(f"{'='*70}\n")

    if args.json:
        print(json.dumps(results, indent=2, default=float))


# ---------------------------------------------------------------------------
# Dashboard command
# ---------------------------------------------------------------------------

def cmd_dashboard(args):
    """Generate full interactive dashboard."""
    from option_pricer.dashboard import full_dashboard

    S = args.S
    K = args.K
    T = args.T
    r = args.r
    sigma = args.sigma or 0.2
    call = not args.put
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'output')

    print(f"Generating interactive dashboard...")
    print(f"  S={S}, K={K}, T={T}, r={r}, sigma={sigma}")

    figs = full_dashboard(S, K, T, r, sigma, call, output_dir=output_dir)

    print(f"\nDashboard generated! Files saved to: {output_dir}")
    for name in figs:
        print(f"  - {name}.html")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Model Options Pricing Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pricing_cli.py price --model bs --S 100 --K 105 --T 0.5 --r 0.03 --sigma 0.2
  python pricing_cli.py price --model heston --S 100 --K 105 --T 0.5 --r 0.03
  python pricing_cli.py greeks --verify --S 100 --K 105 --T 0.5 --r 0.03 --sigma 0.2
  python pricing_cli.py iv --price 3.5 --S 100 --K 105 --T 0.5 --r 0.03
  python pricing_cli.py surface --ticker 510050 --plot
  python pricing_cli.py compare --models bs,heston,merton --S 100 --K 105 --T 0.3
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # ---- Common option parameters ----
    def add_option_args(p):
        p.add_argument('--S', type=float, default=100.0, help='Spot price')
        p.add_argument('--K', type=float, default=105.0, help='Strike price')
        p.add_argument('--T', type=float, default=0.5,
                       help='Time to maturity (years)')
        p.add_argument('--r', type=float, default=0.03, help='Risk-free rate')
        p.add_argument('--sigma', type=float, default=None,
                       help='Volatility')
        p.add_argument('--q', type=float, default=0.0, help='Dividend yield')
        p.add_argument('--put', action='store_true', help='Put option (default: call)')
        p.add_argument('--json', action='store_true', help='Output as JSON')

    # ---- price ----
    price_parser = subparsers.add_parser('price', help='Price an option')
    price_parser.add_argument('--model', type=str, required=True,
                              choices=['bs', 'blackscholes', 'tree', 'binomial',
                                       'mc', 'montecarlo', 'heston', 'merton'],
                              help='Pricing model')
    add_option_args(price_parser)
    price_parser.add_argument('--N', type=int, default=200,
                              help='Steps/paths for tree/mc models')
    price_parser.add_argument('--style', type=str, default='european',
                              choices=['european', 'american'],
                              help='Option style for binomial tree')
    price_parser.add_argument('--no-antithetic', action='store_true',
                              help='Disable antithetic variates for MC')
    price_parser.add_argument('--control-variate', action='store_true',
                              help='Enable control variate for MC')
    # Heston params
    price_parser.add_argument('--v0', type=float, help='Initial variance')
    price_parser.add_argument('--kappa', type=float, default=2.0,
                              help='Mean reversion speed')
    price_parser.add_argument('--theta-v', type=float,
                              help='Long-run variance')
    price_parser.add_argument('--sigma-v', type=float, default=0.3,
                              help='Vol-of-vol')
    price_parser.add_argument('--rho-v', type=float, default=-0.7,
                              help='Asset-variance correlation')
    # Merton params
    price_parser.add_argument('--lam', type=float, default=1.0,
                              help='Jump intensity')
    price_parser.add_argument('--mu-j', type=float, default=-0.05,
                              help='Mean jump size')
    price_parser.add_argument('--sigma-j', type=float, default=0.1,
                              help='Jump size std dev')

    # ---- greeks ----
    greeks_parser = subparsers.add_parser('greeks', help='Compute Greeks')
    greeks_parser.add_argument('--model', type=str, default='bs',
                               choices=['bs', 'blackscholes'],
                               help='Pricing model (for verification)')
    add_option_args(greeks_parser)
    greeks_parser.add_argument('--verify', action='store_true',
                               help='Verify complex-step vs analytic BS Greeks')

    # ---- iv ----
    iv_parser = subparsers.add_parser('iv', help='Compute implied volatility')
    iv_parser.add_argument('--price', type=float, required=True,
                           help='Market option price')
    add_option_args(iv_parser)

    # ---- surface ----
    surface_parser = subparsers.add_parser('surface',
                                            help='Build volatility surface')
    surface_parser.add_argument('--ticker', type=str, default='510050',
                                help='Ticker symbol (e.g. 510050 for SSE 50 ETF)')
    surface_parser.add_argument('--S', type=float, default=None,
                                help='Spot price (auto-detected if not provided)')
    surface_parser.add_argument('--r', type=float, default=0.03,
                                help='Risk-free rate')
    surface_parser.add_argument('--plot', action='store_true',
                                help='Generate Plotly charts')
    surface_parser.add_argument('--output-dir', type=str,
                                help='Output directory for charts')

    # ---- compare ----
    compare_parser = subparsers.add_parser('compare',
                                            help='Compare models')
    compare_parser.add_argument('--models', type=str, required=True,
                                help='Comma-separated models: bs,tree,mc,heston,merton')
    add_option_args(compare_parser)

    # ---- dashboard ----
    dash_parser = subparsers.add_parser('dashboard',
                                         help='Generate full dashboard')
    add_option_args(dash_parser)
    dash_parser.add_argument('--output-dir', type=str,
                             help='Output directory for HTML files')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Route to command
    commands = {
        'price': cmd_price,
        'greeks': cmd_greeks,
        'iv': cmd_iv,
        'surface': cmd_surface,
        'compare': cmd_compare,
        'dashboard': cmd_dashboard,
    }

    commands[args.command](args)


if __name__ == '__main__':
    main()
