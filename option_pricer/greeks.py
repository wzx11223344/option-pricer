"""
Automatic Greeks via Complex-Step Differentiation
==================================================
Compute option Greeks automatically from ANY pricing model function
using complex-step differentiation — no manual formulas needed.

Key insight: f'(x) = Im[f(x + ih)] / h + O(h^2)
This avoids the subtractive cancellation that plagues finite differences.

Reference: Martins, Sturdza, Alonso (2003), "The Complex-Step Derivative Approximation"
"""

import numpy as np
from option_pricer.models import BlackScholes


# ---------------------------------------------------------------------------
# Complex-step derivative core
# ---------------------------------------------------------------------------

def complex_step_first(model_fn, x, h=1e-8, **kwargs):
    """First derivative via complex-step differentiation.

    f'(x) = Im[f(x + ih)] / h + O(h^2)

    Args:
        model_fn: Pricing function, e.g. BlackScholes.call_price(S, K, T, r, sigma)
        x: Parameter value at which to differentiate
        h: Step size (default 1e-8)
        **kwargs: All other parameters for model_fn
    Returns:
        First derivative df/dx
    """
    x_complex = x + 1j * h

    # Build the full argument list, replacing x in the right position
    # We use the model_fn signature to determine where x goes
    # For flexibility, we accept a wrapper approach
    return np.imag(model_fn(x_complex, **kwargs)) / h


def complex_step_second(model_fn, x, h=1e-4, **kwargs):
    """Second derivative via complex-step differentiation.

    f''(x) = 2 * (f(x) - Re[f(x + ih)]) / h^2 + O(h^2)

    Derived from the Taylor expansion of f(x + ih):
    f(x + ih) = f(x) + ih f'(x) - h^2/2 f''(x) - ih^3/6 f'''(x) + h^4/24 f''''(x) + ...
    Re[f(x + ih)] = f(x) - h^2/2 f''(x) + O(h^4)

    Args:
        model_fn: Pricing function
        x: Parameter value
        h: Step size (larger than for first derivative due to h^2 in denominator)
        **kwargs: All other parameters
    Returns:
        Second derivative d^2f/dx^2
    """
    x_complex = x + 1j * h
    f_x = model_fn(x, **kwargs)
    f_complex = model_fn(x_complex, **kwargs)
    return 2.0 * (f_x - np.real(f_complex)) / (h * h)


# ---------------------------------------------------------------------------
# Greek functions — wrap model functions with fixed parameter positions
# ---------------------------------------------------------------------------

def delta(pricing_fn, S, h=1e-8, **params):
    """Delta: d(price)/dS via complex-step differentiation.

    Args:
        pricing_fn: Function f(S, ...) returning option price
        S: Current spot price
        h: Step size
        **params: All other model parameters (K, T, r, sigma, ...)
    Returns:
        Delta value
    """
    return np.imag(pricing_fn(S + 1j * h, **params)) / h


def gamma(pricing_fn, S, h=1e-3, **params):
    """Gamma: d^2(price)/dS^2 via complex-step differentiation.

    Args:
        pricing_fn: Function f(S, ...) returning option price
        S: Current spot price
        h: Step size (larger for 2nd derivative)
        **params: All other model parameters (K, T, r, sigma, ...)
    Returns:
        Gamma value
    """
    f_S = pricing_fn(S, **params)
    f_complex = pricing_fn(S + 1j * h, **params)
    return 2.0 * (f_S - np.real(f_complex)) / (h * h)


def vega(pricing_fn, sigma, h=1e-8, **params):
    """Vega: d(price)/d(sigma) via complex-step differentiation.

    NOTE: vega is conventionally defined as d(price)/d(sigma),
    i.e. the sensitivity to a 1-unit (100%) change in volatility.
    Market convention divides by 100 for a 1% change.

    Args:
        pricing_fn: Function f(..., sigma, ...) returning option price
        sigma: Current volatility
        h: Step size
        **params: All other model parameters
    Returns:
        Vega (price change per unit vol change)
    """
    return np.imag(pricing_fn(sigma=sigma + 1j * h, **params)) / h


def theta(pricing_fn, T, h=1e-8, **params):
    """Theta: -d(price)/dT via complex-step differentiation.

    Args:
        pricing_fn: Function f(..., T, ...) returning option price
        T: Current time to maturity
        h: Step size
        **params: All other model parameters
    Returns:
        Theta (negative time decay, per year)
    """
    deriv = np.imag(pricing_fn(T=T + 1j * h, **params)) / h
    return -deriv  # theta = -dPrice/dT


def rho_greek(pricing_fn, r, h=1e-8, **params):
    """Rho: d(price)/d(r) via complex-step differentiation.

    Args:
        pricing_fn: Function f(..., r, ...) returning option price
        r: Current risk-free rate
        h: Step size
        **params: All other model parameters
    Returns:
        Rho (price change per unit rate change). Divide by 100 for 1% change.
    """
    return np.imag(pricing_fn(r=r + 1j * h, **params)) / h


# ---------------------------------------------------------------------------
# BS model wrappers for complex-step (fixed function signatures)
# ---------------------------------------------------------------------------

def _bs_call_wrapper_S(S, K, T, r, sigma, q=0.0):
    """BS call with S as first arg for complex-step differentiation."""
    T_r = T.real if isinstance(T, complex) else T
    if T_r <= 0:
        S_r = np.real(S) if isinstance(S, complex) else S
        return max(S_r - K, 0.0)
    return BlackScholes.call_price(S, K, T, r, sigma, q)


def _bs_call_wrapper_sigma(sigma, S, K, T, r, q=0.0):
    """BS call with sigma as first arg."""
    T_r = T.real if isinstance(T, complex) else T
    if T_r <= 0:
        return max(S - K, 0.0)
    return BlackScholes.call_price(S, K, T, r, sigma, q)


def _bs_call_wrapper_T(T, S, K, r, sigma, q=0.0):
    """BS call with T as first arg for complex-step differentiation."""
    T_r = T.real if isinstance(T, complex) else T
    if T_r <= 0:
        return max(S - K, 0.0)
    return BlackScholes.call_price(S, K, T, r, sigma, q)


def _bs_call_wrapper_r(r, S, K, T, sigma, q=0.0):
    """BS call with r as first arg."""
    T_r = T.real if isinstance(T, complex) else T
    if T_r <= 0:
        return max(S - K, 0.0)
    return BlackScholes.call_price(S, K, T, r, sigma, q)


def _bs_put_wrapper_S(S, K, T, r, sigma, q=0.0):
    """BS put with S as first arg."""
    T_r = T.real if isinstance(T, complex) else T
    if T_r <= 0:
        S_r = np.real(S) if isinstance(S, complex) else S
        return max(K - S_r, 0.0)
    return BlackScholes.put_price(S, K, T, r, sigma, q)


def _bs_put_wrapper_sigma(sigma, S, K, T, r, q=0.0):
    """BS put with sigma as first arg."""
    T_r = T.real if isinstance(T, complex) else T
    if T_r <= 0:
        return max(K - S, 0.0)
    return BlackScholes.put_price(S, K, T, r, sigma, q)


def _bs_put_wrapper_T(T, S, K, r, sigma, q=0.0):
    """BS put with T as first arg."""
    T_r = T.real if isinstance(T, complex) else T
    if T_r <= 0:
        return max(K - S, 0.0)
    return BlackScholes.put_price(S, K, T, r, sigma, q)


def _bs_put_wrapper_r(r, S, K, T, sigma, q=0.0):
    """BS put with r as first arg."""
    T_r = T.real if isinstance(T, complex) else T
    if T_r <= 0:
        return max(K - S, 0.0)
    return BlackScholes.put_price(S, K, T, r, sigma, q)


# ---------------------------------------------------------------------------
# High-level API: compute all Greeks for a model
# ---------------------------------------------------------------------------

def all_greeks(model='bs', call=True, **params):
    """Compute all Greeks for a given pricing model.

    Args:
        model: 'bs' for Black-Scholes (others use BS as baseline)
        call: True for call option
        **params: Must include S, K, T, r, sigma (and q optionally)
    Returns:
        dict with keys: delta, gamma, vega, theta, rho
    """
    S = params.get('S', 100.0)
    K = params.get('K', 105.0)
    T = params.get('T', 0.5)
    r = params.get('r', 0.03)
    sigma = params.get('sigma', 0.2)
    q = params.get('q', 0.0)

    if model == 'bs':
        # Use analytic formulas for verification
        greeks = {
            'delta': BlackScholes.delta(S, K, T, r, sigma, q, call),
            'gamma': BlackScholes.gamma(S, K, T, r, sigma, q),
            'vega': BlackScholes.vega(S, K, T, r, sigma, q),
            'theta': BlackScholes.theta(S, K, T, r, sigma, q, call),
            'rho': BlackScholes.rho(S, K, T, r, sigma, q, call),
        }
    else:
        # For non-BS models, use complex-step on the model's price function
        # This is the key innovation: model-agnostic Greeks
        if call:
            pricing_fn = lambda S_val, **kw: BlackScholes.call_price(
                S_val, kw.get('K', K), kw.get('T', T),
                kw.get('r', r), kw.get('sigma', sigma), kw.get('q', q))
        else:
            pricing_fn = lambda S_val, **kw: BlackScholes.put_price(
                S_val, kw.get('K', K), kw.get('T', T),
                kw.get('r', r), kw.get('sigma', sigma), kw.get('q', q))

        # Use complex-step via wrapper functions
        greeks = {
            'delta': delta(_bs_call_wrapper_S if call else _bs_put_wrapper_S,
                           S, K=K, T=T, r=r, sigma=sigma, q=q),
            'gamma': gamma(_bs_call_wrapper_S if call else _bs_put_wrapper_S,
                           S, K=K, T=T, r=r, sigma=sigma, q=q),
            'vega': vega(_bs_call_wrapper_sigma if call else _bs_put_wrapper_sigma,
                         sigma, S=S, K=K, T=T, r=r, q=q),
            'theta': theta(_bs_call_wrapper_T if call else _bs_put_wrapper_T,
                           T, S=S, K=K, r=r, sigma=sigma, q=q),
            'rho': rho_greek(_bs_call_wrapper_r if call else _bs_put_wrapper_r,
                             r, S=S, K=K, T=T, sigma=sigma, q=q),
        }
    return greeks


def compute_cs_greeks(S, K, T, r, sigma, q=0.0, call=True):
    """Compute all Greeks for Black-Scholes using complex-step differentiation.

    This serves as a verification: compare against analytic BS Greeks to
    validate the complex-step implementation.

    Args:
        S, K, T, r, sigma, q: BS parameters
        call: True for call option
    Returns:
        dict with 'delta', 'gamma', 'vega', 'theta', 'rho'
    """
    if call:
        cs_delta = delta(_bs_call_wrapper_S, S, K=K, T=T, r=r, sigma=sigma, q=q)
        cs_gamma = gamma(_bs_call_wrapper_S, S, K=K, T=T, r=r, sigma=sigma, q=q)
        cs_vega = vega(_bs_call_wrapper_sigma, sigma, S=S, K=K, T=T, r=r, q=q)
        cs_theta = theta(_bs_call_wrapper_T, T, S=S, K=K, r=r, sigma=sigma, q=q)
        cs_rho = rho_greek(_bs_call_wrapper_r, r, S=S, K=K, T=T, sigma=sigma, q=q)
    else:
        cs_delta = delta(_bs_put_wrapper_S, S, K=K, T=T, r=r, sigma=sigma, q=q)
        cs_gamma = gamma(_bs_put_wrapper_S, S, K=K, T=T, r=r, sigma=sigma, q=q)
        cs_vega = vega(_bs_put_wrapper_sigma, sigma, S=S, K=K, T=T, r=r, q=q)
        cs_theta = theta(_bs_put_wrapper_T, T, S=S, K=K, r=r, sigma=sigma, q=q)
        cs_rho = rho_greek(_bs_put_wrapper_r, r, S=S, K=K, T=T, sigma=sigma, q=q)

    return {
        'delta': cs_delta,
        'gamma': cs_gamma,
        'vega': cs_vega,
        'theta': cs_theta,
        'rho': cs_rho,
    }


def verify_greeks(S=100.0, K=105.0, T=0.5, r=0.03, sigma=0.2, q=0.0):
    """Verify complex-step Greeks against analytic Black-Scholes Greeks.

    Prints a comparison table showing both methods and their differences.

    Returns:
        dict with 'analytic', 'complex_step', 'abs_error', 'rel_error'
    """
    # Analytic Greeks
    analytic = {
        'delta': BlackScholes.delta(S, K, T, r, sigma, q, call=True),
        'gamma': BlackScholes.gamma(S, K, T, r, sigma, q),
        'vega': BlackScholes.vega(S, K, T, r, sigma, q),
        'theta': BlackScholes.theta(S, K, T, r, sigma, q, call=True),
        'rho': BlackScholes.rho(S, K, T, r, sigma, q, call=True),
    }

    # Complex-step Greeks
    complex_step = compute_cs_greeks(S, K, T, r, sigma, q, call=True)

    abs_error = {k: abs(analytic[k] - complex_step[k]) for k in analytic}
    rel_error = {k: abs_error[k] / (abs(analytic[k]) + 1e-12) for k in analytic}

    return {
        'analytic': analytic,
        'complex_step': complex_step,
        'abs_error': abs_error,
        'rel_error': rel_error,
    }
