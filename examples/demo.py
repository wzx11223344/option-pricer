#!/usr/bin/env python
"""
Options Pricing Demo
=====================
Demonstrates all 5 pricing models, automatic Greeks via complex-step
differentiation, volatility surface construction, and interactive dashboards.
"""

import sys
import os
import numpy as np

# Add parent directory to path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from option_pricer.models import (
    BlackScholes, BinomialTree, MonteCarlo, Heston, Merton,
)
from option_pricer.greeks import compute_cs_greeks, verify_greeks, all_greeks
from option_pricer.vol_surface import VolSurface, synthetic_surface

# ============================================================================
# Demo 1: Black-Scholes Pricing
# ============================================================================
def demo_bs():
    print("=" * 70)
    print("  1. BLACK-SCHOLES (1973) - Analytic European Options")
    print("=" * 70)

    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.03, 0.20

    call = BlackScholes.call_price(S, K, T, r, sigma)
    put = BlackScholes.put_price(S, K, T, r, sigma)
    iv = BlackScholes.implied_vol(call, S, K, T, r)

    print(f"\n  Parameters: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"  Call Price:  {call:.6f}")
    print(f"  Put Price:   {put:.6f}")
    print(f"  IV from Call: {iv:.4f} ({iv*100:.2f}%)  [should match sigma]")

    # Put-call parity check
    parity_lhs = call - put
    parity_rhs = S - K * np.exp(-r * T)
    print(f"\n  Put-Call Parity Check:")
    print(f"  C - P = {parity_lhs:.6f}")
    print(f"  S - Ke{{-rT}} = {parity_rhs:.6f}")
    print(f"  Difference: {abs(parity_lhs - parity_rhs):.2e}")
    print()


# ============================================================================
# Demo 2: Binomial Tree
# ============================================================================
def demo_binomial():
    print("=" * 70)
    print("  2. BINOMIAL TREE (CRR 1979) - European & American")
    print("=" * 70)

    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.03, 0.20

    bs_call = BlackScholes.call_price(S, K, T, r, sigma)
    bs_put = BlackScholes.put_price(S, K, T, r, sigma)

    print(f"\n  Convergence Study (N steps → BS limit):")
    print(f"  {'N':>6}  {'Euro Call':>14}  {'Euro Put':>14}  "
          f"{'Amer Put':>14}  {'Early Ex Prem':>14}")
    print(f"  {'-'*70}")

    for N in [10, 50, 100, 200, 500, 1000]:
        euro_call = BinomialTree.price(S, K, T, r, sigma, N=N, call=True, style='european')
        euro_put = BinomialTree.price(S, K, T, r, sigma, N=N, call=False, style='european')
        amer_put = BinomialTree.price(S, K, T, r, sigma, N=N, call=False, style='american')
        eep = amer_put - euro_put

        print(f"  {N:6d}  {euro_call:14.6f}  {euro_put:14.6f}  "
              f"{amer_put:14.6f}  {eep:14.6f}")

    print(f"\n  BS Analytic:  {bs_call:14.6f}  {bs_put:14.6f}")
    print(f"  American put premium > 0 confirms early exercise value.")
    print()


# ============================================================================
# Demo 3: Monte Carlo
# ============================================================================
def demo_mc():
    print("=" * 70)
    print("  3. MONTE CARLO - Variance Reduction Techniques")
    print("=" * 70)

    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.03, 0.20

    bs_call = BlackScholes.call_price(S, K, T, r, sigma)

    # Without antithetic
    price1, se1, ci1_l, ci1_h = MonteCarlo.price(
        S, K, T, r, sigma, N=100000, antithetic=False,
        call=True, control_variate=False, seed=42)

    # With antithetic
    price2, se2, ci2_l, ci2_h = MonteCarlo.price(
        S, K, T, r, sigma, N=100000, antithetic=True,
        call=True, control_variate=False, seed=42)

    # With antithetic + control variate
    price3, se3, ci3_l, ci3_h = MonteCarlo.price(
        S, K, T, r, sigma, N=100000, antithetic=True,
        call=True, control_variate=True, seed=42)

    print(f"\n  BS Analytic Price: {bs_call:.6f}")
    print(f"\n  {'Method':<35} {'Price':>10} {'Std Err':>10} {'95% CI':>30}")
    print(f"  {'-'*85}")
    print(f"  {'Plain MC':<35} {price1:10.6f} {se1:10.6f} "
          f"[{ci1_l:.6f}, {ci1_h:.6f}]")
    print(f"  {'Antithetic Variates':<35} {price2:10.6f} {se2:10.6f} "
          f"[{ci2_l:.6f}, {ci2_h:.6f}]")
    print(f"  {'Antithetic + Control Variate':<35} {price3:10.6f} {se3:10.6f} "
          f"[{ci3_l:.6f}, {ci3_h:.6f}]")

    print(f"\n  Variance reduction: SE reduced from {se1:.6f} to "
          f"{min(se2, se3):.6f} ({se1/min(se2, se3):.1f}x improvement)")
    print()


# ============================================================================
# Demo 4: Heston Stochastic Volatility
# ============================================================================
def demo_heston():
    print("=" * 70)
    print("  4. HESTON (1993) - Stochastic Volatility")
    print("=" * 70)

    S, K, T, r = 100.0, 105.0, 0.5, 0.03
    sigma_bs = 0.20
    bs_call = BlackScholes.call_price(S, K, T, r, sigma_bs)

    # Heston parameters (v0 = initial variance = sigma^2)
    v0 = sigma_bs**2
    params = [
        ('baseline', v0, 2.0, v0, 0.3, -0.7),
        ('high vol-of-vol', v0, 2.0, v0, 0.5, -0.7),
        ('strong neg corr', v0, 2.0, v0, 0.3, -0.95),
        ('zero correlation', v0, 2.0, v0, 0.3, 0.0),
    ]

    print(f"\n  BS Price (sigma={sigma_bs}): {bs_call:.6f}")
    print(f"\n  {'Scenario':<25} {'Price':>10} {'vs BS':>10}")
    print(f"  {'-'*45}")

    for name, v0_, kappa, theta, sigma_v, rho in params:
        h_price = Heston.price(S, K, T, r, v0_, kappa, theta, sigma_v, rho,
                               call=True)
        diff = h_price - bs_call
        print(f"  {name:<25} {h_price:10.6f} {diff:+10.6f}")

    # Calibration demo (synthetic)
    print(f"\n  Calibration Demo:")
    K_range = np.array([80, 90, 95, 100, 105, 110, 120])
    T_vals = np.full_like(K_range, 0.5)
    true_params = {'v0': 0.04, 'kappa': 2.0, 'theta': 0.04,
                   'sigma_v': 0.3, 'rho': -0.7}
    synth_prices = np.array([
        Heston.price(S, k, t, r, **true_params, call=(k >= S))
        for k, t in zip(K_range, T_vals)
    ])

    result = Heston.calibrate(synth_prices, K_range, T_vals, S, r)
    print(f"  True params:  {true_params}")
    print(f"  Calibrated:   {result['params']}")
    print(f"  RMSE: {result['rmse']:.4e} | Success: {result['success']}")
    print()


# ============================================================================
# Demo 5: Merton Jump-Diffusion
# ============================================================================
def demo_merton():
    print("=" * 70)
    print("  5. MERTON (1976) - Jump-Diffusion")
    print("=" * 70)

    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.03, 0.20

    bs_call = BlackScholes.call_price(S, K, T, r, sigma)

    jump_configs = [
        ('no jumps (BS limit)', sigma, 0.0, 0.0, 0.1),
        ('infrequent small jumps', sigma, 0.5, -0.02, 0.05),
        ('frequent small jumps', sigma, 3.0, -0.02, 0.05),
        ('crash risk', sigma, 1.0, -0.10, 0.08),
        ('positive jumps', sigma, 1.0, 0.05, 0.10),
    ]

    print(f"\n  BS Price (sigma={sigma}): {bs_call:.6f}")
    print(f"\n  {'Scenario':<30} {'Price':>10} {'vs BS':>10}")
    print(f"  {'-'*50}")

    for name, sig, lam, mu_j, sigma_j in jump_configs:
        m_price = Merton.price(S, K, T, r, sig, lam, mu_j, sigma_j, call=True)
        diff = m_price - bs_call
        print(f"  {name:<30} {m_price:10.6f} {diff:+10.6f}")

    print()


# ============================================================================
# Demo 6: Automatic Greeks via Complex-Step
# ============================================================================
def demo_greeks():
    print("=" * 70)
    print("  6. AUTOMATIC GREEKS - Complex-Step Differentiation")
    print("=" * 70)

    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.03, 0.20

    # Verify complex-step against analytic BS Greeks
    print(f"\n  Verification: Complex-Step vs Analytic BS Greeks")
    print(f"  S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
    result = verify_greeks(S, K, T, r, sigma)

    print(f"\n  {'Greek':<10} {'Analytic':>14} {'Complex-Step':>14} "
          f"{'Abs Error':>14} {'Rel Error':>14}")
    print(f"  {'-'*66}")

    for key in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        a = result['analytic'][key]
        c = result['complex_step'][key]
        ae = result['abs_error'][key]
        re = result['rel_error'][key]
        print(f"  {key:<10} {a:14.8f} {c:14.8f} {ae:14.2e} {re:14.2e}")

    # Greeks for a put option
    print(f"\n  Put Option Greeks (Complex-Step):")
    put_greeks = compute_cs_greeks(S, K, T, r, sigma, call=False)
    for k, v in put_greeks.items():
        print(f"  {k:<10} {v:14.8f}")

    print(f"\n  Key advantage: No manual derivative formulas needed.")
    print(f"  Works with ANY pricing function — just pass it to delta(), gamma(), etc.")
    print()


# ============================================================================
# Demo 7: Volatility Surface
# ============================================================================
def demo_vol_surface():
    print("=" * 70)
    print("  7. VOLATILITY SURFACE - Construction & Analysis")
    print("=" * 70)

    surf = synthetic_surface(S=100.0, r=0.03)

    # Smile analysis
    smile = surf.smile_analysis()
    print(f"\n  Smile/Skew Analysis per Maturity:")
    print(f"  {'T':>8}  {'ATM IV':>10}  {'Skew':>10}  {'Curvature':>10}  "
          f"{'Pattern'}")
    print(f"  {'-'*60}")
    for T_val, info in sorted(smile.items()):
        print(f"  {T_val:8.4f}  {info['atm_iv']:10.4f}  {info['skew']:10.4f}  "
              f"{info['curvature']:10.4f}  {info['pattern']}")

    # Term structure
    Ts, atm_ivs = surf.term_structure()
    print(f"\n  Term Structure (ATM IV vs Maturity):")
    for t, iv in zip(Ts, atm_ivs):
        print(f"    T={t:.4f}: {iv:.4f} ({iv*100:.2f}%)")

    # Single point lookup
    iv_atm = surf.get_iv(100.0, 0.5)
    iv_otm = surf.get_iv(90.0, 0.5)
    iv_itm = surf.get_iv(110.0, 0.5)
    print(f"\n  Interpolated IVs at T=0.5:")
    print(f"    ATM (K=100):  {iv_atm:.4f} ({iv_atm*100:.2f}%)")
    print(f"    OTM (K=90):   {iv_otm:.4f} ({iv_otm*100:.2f}%)")
    print(f"    ITM (K=110):  {iv_itm:.4f} ({iv_itm*100:.2f}%)")
    print()


# ============================================================================
# Demo 8: Model Comparison
# ============================================================================
def demo_compare():
    print("=" * 70)
    print("  8. CROSS-MODEL COMPARISON")
    print("=" * 70)

    S, K, T, r, sigma = 100.0, 105.0, 0.3, 0.03, 0.20

    models = {
        'Black-Scholes': BlackScholes.call_price(S, K, T, r, sigma),
        'Binomial (N=500)': BinomialTree.price(S, K, T, r, sigma, N=500),
        'Monte Carlo': MonteCarlo.price(S, K, T, r, sigma, N=100000,
                                          antithetic=True, call=True,
                                          control_variate=False, seed=42)[0],
        'Heston': Heston.price(S, K, T, r, v0=sigma**2, kappa=2.0,
                                theta=sigma**2, sigma_v=0.3, rho=-0.7),
        'Merton': Merton.price(S, K, T, r, sigma*0.8, lam=1.0,
                                mu_j=-0.05, sigma_j=0.1),
    }

    bs_price = models['Black-Scholes']
    print(f"\n  {'Model':<25} {'Call Price':>12} {'vs BS':>12} {'Diff %':>10}")
    print(f"  {'-'*60}")

    for name, price in models.items():
        diff = price - bs_price
        diff_pct = (price / bs_price - 1) * 100 if bs_price > 0 else 0
        print(f"  {name:<25} {price:12.6f} {diff:+12.6f} {diff_pct:+9.2f}%")

    print(f"\n  Observations:")
    print(f"  - Binomial converges to BS as N increases")
    print(f"  - Heston adds volatility-of-volatility and correlation effects")
    print(f"  - Merton jump-diffusion adds fat-tail risk premium")
    print(f"  - Monte Carlo converges probabilistically to BS")
    print()


# ============================================================================
# Main
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("  MULTI-MODEL OPTIONS PRICING ENGINE - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("  Black-Scholes | Binomial Tree | Monte Carlo | Heston | Merton")
    print("  Automatic Greeks via Complex-Step Differentiation")
    print("=" * 70)

    demo_bs()
    demo_binomial()
    demo_mc()
    demo_heston()
    demo_merton()
    demo_greeks()
    demo_vol_surface()
    demo_compare()

    print("=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)
    print("\n  Try the CLI:")
    print("    python pricing_cli.py price --model bs --S 100 --K 105 "
          "--T 0.5 --r 0.03 --sigma 0.2")
    print("    python pricing_cli.py greeks --verify --S 100 --K 105 "
          "--T 0.5 --r 0.03 --sigma 0.2")
    print("    python pricing_cli.py compare --models bs,heston,merton "
          "--S 100 --K 105 --T 0.3")
    print()


if __name__ == '__main__':
    main()
