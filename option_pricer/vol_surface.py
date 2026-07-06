"""
Volatility Surface Construction & Analysis
===========================================
Build implied volatility surfaces from option market data.
Supports cubic spline interpolation, smile/skew analysis,
and term structure extraction.

Pure NumPy/SciPy implementation.
"""

import numpy as np
from scipy.interpolate import griddata, CubicSpline
from scipy.optimize import brentq
from option_pricer.models import BlackScholes

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class VolSurface:
    """Implied volatility surface from option chain data."""

    def __init__(self, strikes, maturities, ivs, S, r, q=0.0):
        """Initialize volatility surface.

        Args:
            strikes: Array of strike prices
            maturities: Array of times to maturity (years)
            ivs: Array of implied volatilities
            S: Spot price
            r: Risk-free rate
            q: Dividend yield
        """
        self.S = S
        self.r = r
        self.q = q
        self.strikes = np.asarray(strikes, dtype=float)
        self.maturities = np.asarray(maturities, dtype=float)
        self.ivs = np.asarray(ivs, dtype=float)

        # Filter out NaN implied vols
        mask = ~np.isnan(self.ivs)
        self.strikes = self.strikes[mask]
        self.maturities = self.maturities[mask]
        self.ivs = self.ivs[mask]

        self._unique_T = np.unique(np.round(self.maturities, 8))
        self._unique_K = None  # Will be set if needed

        self._spline_cache = {}
        self._surface_grid = None

    @classmethod
    def from_market_prices(cls, prices, strikes, maturities, types,
                           S, r, q=0.0):
        """Build volatility surface from market prices by inverting BS.

        Args:
            prices: Array of option prices
            strikes: Array of strike prices
            maturities: Array of times to maturity
            types: Array of option types ('call' or 'put', or booleans)
            S: Spot price
            r: Risk-free rate
            q: Dividend yield
        Returns:
            VolSurface instance
        """
        ivs = []
        for i, (price, K, T, opt_type) in enumerate(
                zip(prices, strikes, maturities, types)):
            if T <= 0:
                ivs.append(np.nan)
                continue
            is_call = opt_type in ('call', 'C', True, 1)
            try:
                iv = BlackScholes.implied_vol(price, S, K, T, r, q, call=is_call)
                ivs.append(iv)
            except Exception:
                ivs.append(np.nan)

        return cls(strikes, maturities, ivs, S, r, q)

    def get_iv(self, strike, maturity):
        """Get interpolated implied volatility at (strike, maturity).

        Uses cubic spline in strike dimension, linear in maturity.
        """
        if len(self._unique_T) < 2 or len(self.strikes) < 3:
            # Fallback to linear 2D interpolation
            points = np.column_stack([self.strikes, self.maturities])
            return float(griddata(points, self.ivs, (strike, maturity),
                                  method='linear', fill_value=np.nan))

        # Interpolate in T first, then in K
        # Find two nearest maturities
        T_sorted = np.sort(self._unique_T)
        if maturity <= T_sorted[0]:
            T_idx = 0
            frac = 0.0
        elif maturity >= T_sorted[-1]:
            T_idx = len(T_sorted) - 2
            frac = 1.0
        else:
            T_idx = np.searchsorted(T_sorted, maturity) - 1
            frac = (maturity - T_sorted[T_idx]) / (T_sorted[T_idx + 1] - T_sorted[T_idx])

        T_idx = max(0, min(len(T_sorted) - 2, T_idx))

        # Get IV for each maturity slice at the requested strike
        ivs_at_T = []
        for offset in [0, 1]:
            t_key = round(T_sorted[T_idx + offset], 8)
            mask = np.abs(self.maturities - T_sorted[T_idx + offset]) < 0.001
            K_slice = self.strikes[mask]
            iv_slice = self.ivs[mask]

            if len(K_slice) < 3:
                return float(griddata(points, self.ivs, (strike, maturity),
                                      method='linear', fill_value=np.nan))

            # Sort by strike
            sort_idx = np.argsort(K_slice)
            K_slice = K_slice[sort_idx]
            iv_slice = iv_slice[sort_idx]

            # Cubic spline in strike
            try:
                cs = CubicSpline(K_slice, iv_slice, extrapolate=True)
                iv_at_K = float(cs(strike))
            except Exception:
                iv_at_K = float(np.interp(strike, K_slice, iv_slice))

            ivs_at_T.append(iv_at_K)

        return ivs_at_T[0] * (1 - frac) + ivs_at_T[1] * frac

    def build_grid(self, n_strikes=50, n_maturities=50):
        """Build a regular grid of implied volatilities for surface plotting.

        Returns:
            (K_grid, T_grid, IV_grid)
        """
        K_min, K_max = self.strikes.min(), self.strikes.max()
        K_grid = np.linspace(K_min * 0.9, K_max * 1.1, n_strikes)
        T_grid = np.linspace(
            max(self.maturities.min(), 1/365),
            self.maturities.max() * 1.1, n_maturities)

        K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
        IV_grid = np.zeros_like(K_mesh)

        for i in range(n_maturities):
            for j in range(n_strikes):
                IV_grid[i, j] = self.get_iv(K_mesh[i, j], T_mesh[i, j])

        self._surface_grid = (K_grid, T_grid, IV_grid)
        return self._surface_grid

    def smile_analysis(self):
        """Analyze volatility smile/smirk/skew patterns.

        Returns:
            dict with analysis results per maturity slice.
        """
        results = {}
        for T in self._unique_T:
            mask = np.abs(self.maturities - T) < 0.001
            K_slice = self.strikes[mask]
            iv_slice = self.ivs[mask]

            if len(K_slice) < 3:
                continue

            sort_idx = np.argsort(K_slice)
            K_sorted = K_slice[sort_idx]
            iv_sorted = iv_slice[sort_idx]

            moneyness = K_sorted / self.S
            atm_idx = np.argmin(np.abs(moneyness - 1.0))

            # Skew: slope of IV vs moneyness
            if len(K_sorted) >= 2:
                skew = np.polyfit(moneyness, iv_sorted, 1)[0]
            else:
                skew = 0.0

            # Smile curvature
            if len(K_sorted) >= 3:
                curve = np.polyfit(moneyness, iv_sorted, 2)[0]
            else:
                curve = 0.0

            # ATM IV
            atm_iv = iv_sorted[atm_idx]

            # Pattern classification
            if abs(skew) < 0.02:
                pattern = 'flat'
            elif skew < -0.05:
                pattern = 'smirk (negative skew)'
            elif skew > 0.05:
                pattern = 'reverse smirk (positive skew)'
            else:
                pattern = 'slight skew'

            if curve > 0.1:
                pattern += ' with smile'

            results[round(T, 4)] = {
                'atm_iv': atm_iv,
                'skew': skew,
                'curvature': curve,
                'pattern': pattern,
            }

        return results

    def term_structure(self):
        """Extract ATM implied volatility term structure.

        Returns:
            (maturities, atm_ivs)
        """
        Ts = []
        atm_ivs = []
        for T in self._unique_T:
            mask = np.abs(self.maturities - T) < 0.001
            K_slice = self.strikes[mask]
            iv_slice = self.ivs[mask]

            if len(K_slice) == 0:
                continue

            atm_iv = self.get_iv(self.S, T)
            Ts.append(T)
            atm_ivs.append(atm_iv)

        return np.array(Ts), np.array(atm_ivs)

    def plot_surface(self, title="Implied Volatility Surface"):
        """Generate a 3D Plotly surface of the volatility surface.

        Returns:
            plotly.graph_objects.Figure
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for surface plotting. "
                              "Install with: pip install plotly")

        if self._surface_grid is None:
            self.build_grid()

        K_grid, T_grid, IV_grid = self._surface_grid

        fig = go.Figure(data=[
            go.Surface(
                x=K_grid,
                y=T_grid,
                z=IV_grid,
                colorscale='Viridis',
                colorbar=dict(title='Implied Vol'),
            )
        ])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Strike (K)',
                yaxis_title='Maturity (T, years)',
                zaxis_title='Implied Volatility',
            ),
            width=800,
            height=600,
        )

        return fig

    def plot_smile(self):
        """Plot volatility smiles for each maturity.

        Returns:
            plotly.graph_objects.Figure
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required. Install with: pip install plotly")

        fig = go.Figure()

        for T in self._unique_T:
            mask = np.abs(self.maturities - T) < 0.001
            K_slice = self.strikes[mask]
            iv_slice = self.ivs[mask]

            sort_idx = np.argsort(K_slice)
            fig.add_trace(go.Scatter(
                x=K_slice[sort_idx],
                y=iv_slice[sort_idx],
                mode='lines+markers',
                name=f'T={T:.2f}y',
            ))

        fig.update_layout(
            title='Volatility Smile/Skew by Maturity',
            xaxis_title='Strike (K)',
            yaxis_title='Implied Volatility',
            width=800,
            height=500,
        )

        return fig

    def plot_term_structure(self):
        """Plot ATM volatility term structure.

        Returns:
            plotly.graph_objects.Figure
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required. Install with: pip install plotly")

        Ts, atm_ivs = self.term_structure()

        fig = go.Figure(data=[
            go.Scatter(
                x=Ts,
                y=atm_ivs,
                mode='lines+markers',
                line=dict(width=3),
                marker=dict(size=8),
            )
        ])

        fig.update_layout(
            title='ATM Implied Volatility Term Structure',
            xaxis_title='Maturity (years)',
            yaxis_title='ATM Implied Volatility',
            width=800,
            height=400,
        )

        return fig


# ---------------------------------------------------------------------------
# Synthetic surface builder (for demo/testing without market data)
# ---------------------------------------------------------------------------

def synthetic_surface(S=100.0, r=0.03):
    """Create a synthetic volatility surface for demonstration.

    Generates realistic option chain data with volatility smile/skew.
    """
    strikes = []
    maturities = []
    ivs = []

    np.random.seed(42)
    Ts = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    for T in Ts:
        # ATM vol level
        atm_vol = 0.15 + 0.03 * np.sqrt(T)

        # Generate strikes around ATM
        K_range = np.linspace(S * 0.7, S * 1.3, 15)
        for K in K_range:
            moneyness = K / S
            # Skew: lower strikes have higher IV (equity smirk)
            skew_adj = -0.08 * (moneyness - 1.0)
            # Smile curvature
            smile_adj = 0.05 * (moneyness - 1.0)**2
            # Term structure adjustment
            term_adj = 0.02 * (np.sqrt(T) - 0.5)

            iv = atm_vol + skew_adj + smile_adj + term_adj
            iv += np.random.normal(0, 0.003)  # small noise

            iv = max(0.01, min(iv, 2.0))
            strikes.append(K)
            maturities.append(T)
            ivs.append(iv)

    return VolSurface(strikes, maturities, ivs, S, r)
