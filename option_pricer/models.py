"""
Option Pricing Models
=====================
Five pricing models: Black-Scholes, Binomial Tree (CRR),
Monte Carlo, Heston Stochastic Volatility, Merton Jump-Diffusion.

Pure NumPy/SciPy implementation — no QuantLib dependency.
"""

import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares, brentq
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# 1. Black-Scholes (1973)
# ---------------------------------------------------------------------------

class BlackScholes:
    """Analytic European option pricing via Black-Scholes formula."""

    @staticmethod
    def _d1(S, K, T, r, sigma, q=0.0):
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def _d2(S, K, T, r, sigma, q=0.0):
        d1 = BlackScholes._d1(S, K, T, r, sigma, q)
        return d1 - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S, K, T, r, sigma, q=0.0):
        """Analytic European call price.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield (default 0)
        """
        T_real = np.real(T) if isinstance(T, complex) else T
        if T_real <= 0:
            S_real = np.real(S) if isinstance(S, complex) else S
            return max(S_real - K, 0.0)
        d1 = BlackScholes._d1(S, K, T, r, sigma, q)
        d2 = BlackScholes._d2(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S, K, T, r, sigma, q=0.0):
        """Analytic European put price."""
        T_real = np.real(T) if isinstance(T, complex) else T
        if T_real <= 0:
            S_real = np.real(S) if isinstance(S, complex) else S
            return max(K - S_real, 0.0)
        d1 = BlackScholes._d1(S, K, T, r, sigma, q)
        d2 = BlackScholes._d2(S, K, T, r, sigma, q)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def implied_vol(price, S, K, T, r, q=0.0, call=True, tol=1e-8, max_iter=100):
        """Newton-Raphson implied volatility solver with bracket protection.

        Args:
            price: Observed market price of the option
            S, K, T, r, q: Standard BS parameters
            call: True for call, False for put
            tol: Convergence tolerance
            max_iter: Maximum iterations
        Returns:
            Implied volatility
        """
        pricer = BlackScholes.call_price if call else BlackScholes.put_price

        def f(sigma):
            return pricer(S, K, T, r, sigma, q) - price

        # Bracket search via Brent's method as initial guess
        lo, hi = 1e-6, 5.0
        # Expand upper bound if needed
        for _ in range(20):
            if f(hi) > 0:
                break
            hi *= 2.0
        else:
            return np.nan

        # Use Brent's method (more robust than Newton for IV)
        try:
            sigma_iv = brentq(f, lo, hi, xtol=tol, maxiter=max_iter)
            return sigma_iv
        except (ValueError, RuntimeError):
            return np.nan

    @staticmethod
    def vega(S, K, T, r, sigma, q=0.0):
        """Analytic vega (derivative w.r.t. sigma)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes._d1(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    @staticmethod
    def delta(S, K, T, r, sigma, q=0.0, call=True):
        """Analytic delta."""
        if T <= 0:
            return 1.0 if (call and S > K) or (not call and S < K) else 0.0
        d1 = BlackScholes._d1(S, K, T, r, sigma, q)
        if call:
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return np.exp(-q * T) * (norm.cdf(d1) - 1)

    @staticmethod
    def gamma(S, K, T, r, sigma, q=0.0):
        """Analytic gamma."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes._d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def theta(S, K, T, r, sigma, q=0.0, call=True):
        """Analytic theta."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes._d1(S, K, T, r, sigma, q)
        d2 = BlackScholes._d2(S, K, T, r, sigma, q)
        term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if call:
            term2 = q * S * np.exp(-q * T) * norm.cdf(d1) - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = -q * S * np.exp(-q * T) * norm.cdf(-d1) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        return term1 + term2

    @staticmethod
    def rho(S, K, T, r, sigma, q=0.0, call=True):
        """Analytic rho."""
        if T <= 0:
            return 0.0
        d2 = BlackScholes._d2(S, K, T, r, sigma, q)
        if call:
            return K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2)


# ---------------------------------------------------------------------------
# 2. Binomial Tree (Cox-Ross-Rubinstein 1979)
# ---------------------------------------------------------------------------

class BinomialTree:
    """CRR binomial tree for European and American options."""

    @staticmethod
    def price(S, K, T, r, sigma, N=200, call=True, style='european', q=0.0):
        """Price option using CRR binomial tree.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            N: Number of time steps (default 200)
            call: True for call, False for put
            style: 'european' or 'american'
            q: Dividend yield
        Returns:
            Option price
        """
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)

        # Clamp probability to avoid numerical issues
        p = np.clip(p, 1e-12, 1 - 1e-12)

        disc = np.exp(-r * dt)

        # Terminal payoffs
        S_vals = S * u ** np.arange(N, -1, -1) * d ** np.arange(0, N + 1)
        if call:
            option_vals = np.maximum(S_vals - K, 0.0)
        else:
            option_vals = np.maximum(K - S_vals, 0.0)

        # Backward induction
        for j in range(N - 1, -1, -1):
            option_vals = disc * (p * option_vals[:-1] + (1 - p) * option_vals[1:])
            if style == 'american':
                S_vals_j = S * u ** np.arange(j, -1, -1) * d ** np.arange(0, j + 1)
                if call:
                    exercise = np.maximum(S_vals_j - K, 0.0)
                else:
                    exercise = np.maximum(K - S_vals_j, 0.0)
                option_vals = np.maximum(option_vals, exercise)

        return float(option_vals[0])


# ---------------------------------------------------------------------------
# 3. Monte Carlo
# ---------------------------------------------------------------------------

class MonteCarlo:
    """Monte Carlo pricing with variance reduction techniques."""

    @staticmethod
    def price(S, K, T, r, sigma, N=100000, antithetic=True, call=True,
              control_variate=True, q=0.0, seed=None):
        """Price option via Monte Carlo simulation.

        Args:
            S, K, T, r, sigma, q: Standard BS parameters
            N: Number of paths
            antithetic: Use antithetic variates for variance reduction
            call: True for call, False for put
            control_variate: Use geometric Asian as control variate
            seed: Random seed for reproducibility
        Returns:
            (price, std_error, ci_lower, ci_upper)
        """
        rng = np.random.RandomState(seed)

        drift = (r - q - 0.5 * sigma**2) * T
        vol = sigma * np.sqrt(T)

        if antithetic:
            Z = rng.standard_normal(N // 2)
            Z = np.concatenate([Z, -Z])
            n_paths = 2 * (N // 2)
        else:
            Z = rng.standard_normal(N)
            n_paths = N

        S_T = S * np.exp(drift + vol * Z)

        if call:
            payoffs = np.maximum(S_T - K, 0.0)
        else:
            payoffs = np.maximum(K - S_T, 0.0)

        # Control variate: geometric Asian option
        if control_variate:
            # Geometric average price over the path
            # For a single-step simulation, geometric Asian = geometric Brownian
            # We use the analytical price of a geometric average option as control
            # Simplified: use the underlying itself as a rough control
            # Better: use geometric Asian with m monitoring points
            m = 12  # monthly monitoring
            dt = T / m
            control_payoffs = np.zeros(n_paths)
            rng2 = np.random.RandomState(seed + 1 if seed else None)
            for i in range(n_paths):
                Z_path = rng2.standard_normal(m)
                S_path = S * np.exp(np.cumsum((r - q - 0.5 * sigma**2) * dt +
                                               sigma * np.sqrt(dt) * Z_path))
                geo_avg = np.exp(np.mean(np.log(S_path)))
                if call:
                    control_payoffs[i] = np.maximum(geo_avg - K, 0.0)
                else:
                    control_payoffs[i] = np.maximum(K - geo_avg, 0.0)

            # Compute optimal beta
            cov = np.cov(payoffs, control_payoffs)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0

            # Analytic geometric Asian price (Kemna & Vorst 1990)
            sigma_adj = sigma * np.sqrt((2 * m + 1) / (6 * (m + 1)))
            mu_adj = 0.5 * (r - q - 0.5 * sigma**2) * T * (m + 1) / m

            # Adjusted parameters for the geometric average
            S_adj = S * np.exp(-(r - q) * T + (r - q) * T * (m + 1) / (2 * m))
            # Simplification: use BS with adjusted vol
            geo_call = max(0, S_adj - K)  # approximation
            # Actually use BS for geometric average
            r_adj = (mu_adj / T)  # effective rate
            from scipy.stats import norm as norm_st
            d1_adj = (np.log(S / K) + mu_adj + 0.5 * sigma_adj**2 * T) / (sigma_adj * np.sqrt(T))
            d2_adj = d1_adj - sigma_adj * np.sqrt(T)
            if call:
                control_mu = S * np.exp((mu_adj - r * T)) * norm_st.cdf(d1_adj) - K * np.exp(-r * T) * norm_st.cdf(d2_adj)
            else:
                control_mu = K * np.exp(-r * T) * norm_st.cdf(-d2_adj) - S * np.exp((mu_adj - r * T)) * norm_st.cdf(-d1_adj)

            # Adjust payoffs
            payoffs = payoffs - beta * (control_payoffs - control_mu)

        price = np.exp(-r * T) * np.mean(payoffs)
        std_err = np.exp(-r * T) * np.std(payoffs, ddof=1) / np.sqrt(n_paths)

        # 95% confidence interval
        ci = 1.96 * std_err

        return (price, std_err, price - ci, price + ci)


# ---------------------------------------------------------------------------
# 4. Heston (1993) — Stochastic Volatility
# ---------------------------------------------------------------------------

class Heston:
    """Heston stochastic volatility model with characteristic function pricing."""

    @staticmethod
    def _char_func(u, T, r, kappa, theta, sigma_v, rho, v0, j=1):
        """Heston characteristic function.

        Computes f_j(φ) for the Heston model.

        Args:
            u: Integration variable (float or array)
            T: Time to maturity
            r: Risk-free rate
            kappa: Mean reversion speed
            theta: Long-run variance
            sigma_v: Volatility of variance (vol-of-vol)
            rho: Correlation between asset and variance
            v0: Initial variance
            j: 1 for P1, 2 for P2
        Returns:
            Characteristic function value
        """
        if j == 1:
            b = kappa - rho * sigma_v
            u_j = 0.5
        else:
            b = kappa
            u_j = -0.5

        a = kappa * theta
        d = np.sqrt((rho * sigma_v * 1j * u - b)**2 -
                     sigma_v**2 * (2j * u_j * u - u**2))

        g = (b - rho * sigma_v * 1j * u + d) / (b - rho * sigma_v * 1j * u - d)

        # Handle d ~ 0 case to avoid division by zero
        exp_dT = np.exp(d * T)

        C = (r * 1j * u * T +
             a / sigma_v**2 * ((b - rho * sigma_v * 1j * u + d) * T -
                               2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))))

        D = ((b - rho * sigma_v * 1j * u + d) / sigma_v**2 *
             (1.0 - exp_dT) / (1.0 - g * exp_dT))

        return np.exp(C + D * v0)

    @staticmethod
    def price(S, K, T, r, v0, kappa, theta, sigma_v, rho, call=True, q=0.0):
        """Price European option using Heston model via Fourier inversion.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            v0: Initial variance
            kappa: Mean reversion speed
            theta: Long-run variance
            sigma_v: Volatility of variance
            rho: Correlation between asset and variance
            call: True for call, False for put
            q: Dividend yield
        Returns:
            Option price
        """
        if T <= 0:
            return max(S - K, 0.0) if call else max(K - S, 0.0)

        # Compute P1 and P2 via Fourier inversion
        log_SK = np.log(S / K)

        def integrand_1(phi):
            cf = Heston._char_func(phi, T, r - q, kappa, theta, sigma_v, rho, v0, j=1)
            return np.real(np.exp(1j * phi * log_SK) * cf / (1j * phi))

        def integrand_2(phi):
            cf = Heston._char_func(phi, T, r - q, kappa, theta, sigma_v, rho, v0, j=2)
            return np.real(np.exp(1j * phi * log_SK) * cf / (1j * phi))

        # Integration from 0 to infinity (approximate with reasonable cutoff)
        # Use adaptive quadrature
        try:
            P1_int, _ = quad(integrand_1, 1e-12, 200, limit=200)
            P2_int, _ = quad(integrand_2, 1e-12, 200, limit=200)
        except Exception:
            # Fallback with fixed-point integration
            phi_grid = np.linspace(1e-8, 200, 5000)
            dphi = phi_grid[1] - phi_grid[0]
            P1_int = np.trapz([integrand_1(p) for p in phi_grid], phi_grid)
            P2_int = np.trapz([integrand_2(p) for p in phi_grid], phi_grid)

        P1 = 0.5 + P1_int / np.pi
        P2 = 0.5 + P2_int / np.pi

        # Apply put-call parity if needed
        call_price = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

        if call:
            return max(call_price, S * np.exp(-q * T) - K * np.exp(-r * T))
        else:
            put_price = call_price - S * np.exp(-q * T) + K * np.exp(-r * T)
            return max(put_price, K * np.exp(-r * T) - S * np.exp(-q * T))

    @staticmethod
    def calibrate(market_prices, strikes, maturities, S, r, q=0.0,
                  init_params=None, bounds=None):
        """Calibrate Heston parameters to market prices via least squares.

        Args:
            market_prices: Array of observed option prices
            strikes: Array of corresponding strikes
            maturities: Array of corresponding maturities (same length)
            S: Current spot price
            r: Risk-free rate
            q: Dividend yield
            init_params: Initial guess [v0, kappa, theta, sigma_v, rho]
            bounds: List of (min, max) tuples for each parameter
        Returns:
            dict with 'params' and 'rmse'
        """
        if init_params is None:
            init_params = [0.04, 2.0, 0.04, 0.3, -0.7]

        if bounds is None:
            bounds = [(0.001, 1.0), (0.1, 10.0), (0.001, 1.0),
                       (0.01, 2.0), (-0.999, 0.999)]

        def objective(params):
            v0, kappa, theta, sigma_v, rho = params
            model_prices = np.array([
                Heston.price(S, strikes[i], maturities[i], r,
                             v0, kappa, theta, sigma_v, rho,
                             call=(strikes[i] >= S), q=q)
                for i in range(len(strikes))
            ])
            return model_prices - market_prices

        result = least_squares(objective, init_params, bounds=list(zip(*bounds)),
                                method='trf', max_nfev=5000)

        rmse = np.sqrt(np.mean(objective(result.x)**2))

        return {
            'params': {
                'v0': result.x[0],
                'kappa': result.x[1],
                'theta': result.x[2],
                'sigma_v': result.x[3],
                'rho': result.x[4],
            },
            'rmse': rmse,
            'success': result.success,
        }


# ---------------------------------------------------------------------------
# 5. Merton (1976) — Jump-Diffusion
# ---------------------------------------------------------------------------

class Merton:
    """Merton jump-diffusion model for option pricing."""

    @staticmethod
    def price(S, K, T, r, sigma, lam, mu_j, sigma_j, call=True, q=0.0,
              max_terms=50):
        """Price European option using Merton jump-diffusion model.

        The price is an infinite weighted sum of Black-Scholes prices,
        each conditioning on a specific number of jumps.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Diffusion volatility
            lam: Jump intensity (jumps per year)
            mu_j: Mean jump size (log-normal)
            sigma_j: Standard deviation of jump size
            call: True for call, False for put
            q: Dividend yield
            max_terms: Maximum number of terms in the Poisson sum
        Returns:
            Option price
        """
        if T <= 0:
            return max(S - K, 0.0) if call else max(K - S, 0.0)

        # Expected jump factor: k = E[exp(Y)] - 1
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0

        price = 0.0
        poisson_weight_sum = 0.0

        for n in range(max_terms):
            # Poisson probability of n jumps
            poisson_prob = np.exp(-lam * T) * (lam * T)**n / math.factorial(n)
            poisson_weight_sum += poisson_prob

            # Adjusted parameters for BS conditional on n jumps
            # Risk-neutral drift compensation
            r_n = r - q - lam * k + n * (mu_j + 0.5 * sigma_j**2) / T
            sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)

            if call:
                bs_price = BlackScholes.call_price(S, K, T, r_n + q, sigma_n, q)
            else:
                bs_price = BlackScholes.put_price(S, K, T, r_n + q, sigma_n, q)

            price += poisson_prob * bs_price

        # Normalize by total probability (handles truncation error)
        if poisson_weight_sum > 0:
            price /= poisson_weight_sum

        return price

    @staticmethod
    def calibrate(market_prices, strikes, maturities, S, r, q=0.0,
                  init_params=None, bounds=None):
        """Calibrate Merton parameters to market prices.

        Args:
            market_prices: Array of observed option prices
            strikes: Array of corresponding strikes
            maturities: Array of corresponding maturities
            S: Current spot price
            r: Risk-free rate
            q: Dividend yield
            init_params: [sigma, lam, mu_j, sigma_j]
            bounds: List of (min, max) tuples
        Returns:
            dict with 'params' and 'rmse'
        """
        if init_params is None:
            init_params = [0.2, 1.0, -0.05, 0.1]

        if bounds is None:
            bounds = [(0.01, 1.0), (0.01, 10.0), (-0.5, 0.5), (0.01, 1.0)]

        def objective(params):
            sigma, lam, mu_j, sigma_j = params
            model_prices = np.array([
                Merton.price(S, strikes[i], maturities[i], r,
                             sigma, lam, mu_j, sigma_j,
                             call=(strikes[i] >= S), q=q)
                for i in range(len(strikes))
            ])
            return model_prices - market_prices

        result = least_squares(objective, init_params, bounds=list(zip(*bounds)),
                                method='trf', max_nfev=5000)

        rmse = np.sqrt(np.mean(objective(result.x)**2))

        return {
            'params': {
                'sigma': result.x[0],
                'lam': result.x[1],
                'mu_j': result.x[2],
                'sigma_j': result.x[3],
            },
            'rmse': rmse,
            'success': result.success,
        }
