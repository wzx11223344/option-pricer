"""
Interactive Dashboards
======================
Plotly-based interactive visualization dashboards for options analysis.

Charts:
- P&L Heatmap: Option P&L at expiration across S and sigma scenarios
- Greeks Surface: delta/gamma/vega/theta vs S and T
- Model Comparison: overlay 5 model prices for same option
- Time Decay: option value over time (theta bleed)
- Vol Surface 3D: interactive implied volatility surface
"""

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from option_pricer.models import (
    BlackScholes, BinomialTree, MonteCarlo, Heston, Merton
)
from option_pricer.greeks import compute_cs_greeks


def _check_plotly():
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly is required for dashboards. Install with: pip install plotly"
        )


# ---------------------------------------------------------------------------
# 1. P&L Heatmap
# ---------------------------------------------------------------------------

def pnl_heatmap(S0=100.0, K=105.0, T=0.5, r=0.03, sigma=0.2, call=True,
                premium=None):
    """Generate P&L heatmap at expiration across S and sigma scenarios.

    Args:
        S0: Current spot price
        K: Strike
        T: Time to maturity
        r: Risk-free rate
        sigma: Current implied volatility
        call: True for call option
        premium: Option premium paid (defaults to BS price if None)
    Returns:
        plotly.graph_objects.Figure
    """
    _check_plotly()

    if premium is None:
        premium = BlackScholes.call_price(S0, K, T, r, sigma) if call else \
                  BlackScholes.put_price(S0, K, T, r, sigma)

    # Scenario ranges
    S_range = np.linspace(S0 * 0.7, S0 * 1.3, 60)
    sigma_range = np.linspace(max(sigma * 0.3, 0.05), sigma * 2.5, 50)
    S_mesh, sigma_mesh = np.meshgrid(S_range, sigma_range)

    # Compute P&L
    pnl = np.zeros_like(S_mesh)
    for i in range(len(sigma_range)):
        for j in range(len(S_range)):
            if call:
                price = BlackScholes.call_price(S_mesh[i, j], K, T, r, sigma_mesh[i, j])
            else:
                price = BlackScholes.put_price(S_mesh[i, j], K, T, r, sigma_mesh[i, j])
            pnl[i, j] = price - premium

    fig = go.Figure(data=[
        go.Heatmap(
            z=pnl,
            x=S_range,
            y=sigma_range,
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title='P&L'),
            hovertemplate='S: %{x:.1f}<br>sigma: %{y:.3f}<br>P&L: %{z:.3f}<extra></extra>',
        )
    ])

    opt_type = 'Call' if call else 'Put'
    fig.update_layout(
        title=f'{opt_type} Option P&L Heatmap (K={K}, T={T}, premium={premium:.3f})',
        xaxis_title='Spot Price (S)',
        yaxis_title='Volatility (sigma)',
        width=850,
        height=600,
    )

    # Add P&L=0 contour
    fig.add_trace(
        go.Contour(
            z=pnl,
            x=S_range,
            y=sigma_range,
            contours=dict(
                start=0, end=0, size=1,
                coloring='lines',
                showlabels=False,
            ),
            line=dict(width=2, color='black', dash='dash'),
            showscale=False,
            name='P&L = 0',
        )
    )

    # Mark current position
    fig.add_trace(
        go.Scatter(
            x=[S0],
            y=[sigma],
            mode='markers',
            marker=dict(size=12, color='blue', symbol='x'),
            name='Current',
        )
    )

    return fig


# ---------------------------------------------------------------------------
# 2. Greeks Surface
# ---------------------------------------------------------------------------

def greeks_surface(S0=100.0, K=105.0, r=0.03, sigma=0.2, call=True):
    """Generate Greeks surface plots (delta/gamma/vega/theta) vs S and T.

    Returns:
        plotly.graph_objects.Figure
    """
    _check_plotly()

    S_range = np.linspace(S0 * 0.6, S0 * 1.4, 50)
    T_range = np.linspace(1/365, 2.0, 50)
    S_mesh, T_mesh = np.meshgrid(S_range, T_range)

    delta_grid = np.zeros_like(S_mesh)
    gamma_grid = np.zeros_like(S_mesh)
    vega_grid = np.zeros_like(S_mesh)
    theta_grid = np.zeros_like(S_mesh)

    for i in range(len(T_range)):
        for j in range(len(S_range)):
            greeks = compute_cs_greeks(
                S_mesh[i, j], K, T_mesh[i, j], r, sigma, call=call)
            delta_grid[i, j] = greeks['delta']
            gamma_grid[i, j] = greeks['gamma']
            vega_grid[i, j] = greeks['vega']
            theta_grid[i, j] = greeks['theta']

    opt_type = 'Call' if call else 'Put'
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Delta', 'Gamma', 'Vega', 'Theta'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'surface'}, {'type': 'surface'}]],
    )

    surfaces = [
        (delta_grid, 1, 1), (gamma_grid, 1, 2),
        (vega_grid, 2, 1), (theta_grid, 2, 2),
    ]

    for grid, row, col in surfaces:
        fig.add_trace(
            go.Surface(
                z=grid,
                x=S_range,
                y=T_range,
                colorscale='Viridis',
                showscale=False,
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title=f'{opt_type} Option Greeks Surface (K={K}, sigma={sigma})',
        width=1000,
        height=800,
    )

    # Update axis titles
    for i in range(1, 5):
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1
        fig.update_scenes(
            xaxis_title='Spot (S)',
            yaxis_title='Maturity (T)',
            zaxis_title='',
            row=row, col=col,
        )

    return fig


# ---------------------------------------------------------------------------
# 3. Model Comparison
# ---------------------------------------------------------------------------

def model_comparison(S=100.0, K=105.0, T=0.3, r=0.03, sigma=0.2,
                     call=True, N_steps=50):
    """Compare option prices across all 5 models for varying parameters.

    Generates price vs strike and price vs maturity comparison charts.

    Returns:
        plotly.graph_objects.Figure
    """
    _check_plotly()

    # Model configurations
    models_config = {
        'Black-Scholes': lambda S_val, K_val, T_val:
            BlackScholes.call_price(S_val, K_val, T_val, r, sigma) if call else
            BlackScholes.put_price(S_val, K_val, T_val, r, sigma),

        'Binomial Tree': lambda S_val, K_val, T_val:
            BinomialTree.price(S_val, K_val, T_val, r, sigma, N=200, call=call),

        'Monte Carlo': lambda S_val, K_val, T_val:
            MonteCarlo.price(S_val, K_val, T_val, r, sigma,
                             N=50000, antithetic=True, call=call,
                             control_variate=False)[0],

        'Heston': lambda S_val, K_val, T_val:
            Heston.price(S_val, K_val, T_val, r,
                         v0=sigma**2, kappa=2.0, theta=sigma**2,
                         sigma_v=0.3, rho=-0.7, call=call),

        'Merton': lambda S_val, K_val, T_val:
            Merton.price(S_val, K_val, T_val, r,
                         sigma=sigma*0.8, lam=1.0, mu_j=-0.05,
                         sigma_j=0.1, call=call),
    }

    # 1) Price vs Strike
    K_range = np.linspace(S * 0.7, S * 1.3, N_steps)
    fig1 = go.Figure()
    for name, model_fn in models_config.items():
        prices = [model_fn(S, k, T) for k in K_range]
        fig1.add_trace(go.Scatter(
            x=K_range, y=prices, mode='lines',
            name=name, line=dict(width=2),
        ))

    # Mark ATM
    fig1.add_vline(x=S, line_dash='dash', line_color='gray',
                    annotation_text='ATM')
    opt_type = 'Call' if call else 'Put'

    fig1.update_layout(
        title=f'{opt_type} Option - Model Price Comparison (T={T})',
        xaxis_title='Strike (K)',
        yaxis_title='Option Price',
        width=900,
        height=500,
        hovermode='x unified',
    )

    # 2) Price vs Maturity
    T_range = np.linspace(0.05, 2.0, N_steps)
    fig2 = go.Figure()
    for name, model_fn in models_config.items():
        prices = [model_fn(S, K, t) for t in T_range]
        fig2.add_trace(go.Scatter(
            x=T_range, y=prices, mode='lines',
            name=name, line=dict(width=2),
        ))

    fig2.update_layout(
        title=f'{opt_type} Option - Model Price vs Maturity (K={K})',
        xaxis_title='Maturity (years)',
        yaxis_title='Option Price',
        width=900,
        height=500,
        hovermode='x unified',
    )

    return fig1, fig2


# ---------------------------------------------------------------------------
# 4. Time Decay (Theta Bleed)
# ---------------------------------------------------------------------------

def time_decay(S=100.0, K=105.0, r=0.03, sigma=0.2, call=True,
               days_to_expiry=365):
    """Visualize option value decay over time (theta bleed).

    Args:
        days_to_expiry: Number of days to simulate
    Returns:
        plotly.graph_objects.Figure
    """
    _check_plotly()

    days = np.arange(days_to_expiry, 0, -1)
    T_vals = days / 365.0

    prices = []
    deltas = []
    thetas = []

    for T_val in T_vals:
        p = BlackScholes.call_price(S, K, T_val, r, sigma) if call else \
            BlackScholes.put_price(S, K, T_val, r, sigma)
        d = BlackScholes.delta(S, K, T_val, r, sigma, call=call)
        t = BlackScholes.theta(S, K, T_val, r, sigma, call=call)

        prices.append(p)
        deltas.append(d)
        thetas.append(t)

    prices = np.array(prices)
    deltas = np.array(deltas)
    thetas = np.array(thetas)

    opt_type = 'Call' if call else 'Put'

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f'{opt_type} Option Price Decay',
            'Delta Evolution',
            'Theta (Daily Decay)',
        ),
        vertical_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(x=days, y=prices, mode='lines',
                   fill='tozeroy', name='Price',
                   line=dict(width=2, color='blue')),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=days, y=deltas, mode='lines',
                   name='Delta', line=dict(width=2, color='green')),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=days, y=thetas / 365, mode='lines',
                   name='Theta (per day)', line=dict(width=2, color='red')),
        row=3, col=1,
    )

    fig.update_xaxes(title_text='Days to Expiry', row=3, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Delta', row=2, col=1)
    fig.update_yaxes(title_text='Theta/day', row=3, col=1)

    fig.update_layout(
        height=800,
        width=800,
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------------------------
# 5. Vol Surface 3D (standalone, no VolSurface class needed)
# ---------------------------------------------------------------------------

def vol_surface_3d(strikes, maturities, ivs, S=None,
                   title="Implied Volatility Surface"):
    """Create interactive 3D volatility surface from raw data.

    Args:
        strikes: Array of strike prices
        maturities: Array of maturities
        ivs: Array of implied volatilities
        S: Spot price (optional, for ATM line)
        title: Plot title
    Returns:
        plotly.graph_objects.Figure
    """
    _check_plotly()

    fig = go.Figure(data=[
        go.Scatter3d(
            x=strikes,
            y=maturities,
            z=ivs,
            mode='markers',
            marker=dict(
                size=3,
                color=ivs,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title='IV'),
            ),
            name='Market IVs',
        )
    ])

    # Add mesh surface via interpolation
    from scipy.interpolate import griddata

    K_unique = np.linspace(np.min(strikes), np.max(strikes), 40)
    T_unique = np.linspace(np.min(maturities), np.max(maturities), 40)
    K_mesh, T_mesh = np.meshgrid(K_unique, T_unique)

    points = np.column_stack([strikes, maturities])
    IV_mesh = griddata(points, ivs, (K_mesh, T_mesh), method='cubic')

    fig.add_trace(
        go.Surface(
            x=K_unique,
            y=T_unique,
            z=IV_mesh,
            colorscale='Viridis',
            opacity=0.6,
            showscale=False,
            name='Surface',
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Strike (K)',
            yaxis_title='Maturity (T)',
            zaxis_title='Implied Volatility',
        ),
        width=900,
        height=700,
    )

    return fig


# ---------------------------------------------------------------------------
# Master dashboard (all charts)
# ---------------------------------------------------------------------------

def full_dashboard(S=100.0, K=105.0, T=0.5, r=0.03, sigma=0.2, call=True,
                   output_dir=None):
    """Generate and optionally save all dashboard charts.

    Args:
        output_dir: If provided, save all charts as HTML files
    Returns:
        dict of {name: plotly Figure}
    """
    _check_plotly()

    figs = {}

    # P&L Heatmap
    figs['pnl_heatmap'] = pnl_heatmap(S, K, T, r, sigma, call)

    # Greeks surface
    figs['greeks_surface'] = greeks_surface(S, K, r, sigma, call)

    # Model comparison
    fig1, fig2 = model_comparison(S, K, T, r, sigma, call)
    figs['model_comparison_K'] = fig1
    figs['model_comparison_T'] = fig2

    # Time decay
    figs['time_decay'] = time_decay(S, K, r, sigma, call, days_to_expiry=int(T*365))

    # Vol surface (synthetic for demo)
    from option_pricer.vol_surface import synthetic_surface
    surf = synthetic_surface(S, r)
    figs['vol_surface'] = surf.plot_surface()

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        for name, fig in figs.items():
            path = os.path.join(output_dir, f'{name}.html')
            fig.write_html(path)
            print(f"Saved: {path}")

    return figs
