import math
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf

# Default short-term risk-free rates by currency (central bank policy rates)
_DEFAULT_RATES = {
    "USD": 3.6,
    "EUR": 2.0,
    "GBP": 3.75,
    "JPY": 0.75,
    "CHF": 0.0,
}


def get_risk_free_rate(currency="USD"):
    """Fetch the risk-free rate for a given currency.

    For USD, fetches the 13-week T-bill yield (^IRX) from yfinance.
    For other currencies, returns a hardcoded default.
    Falls back to the USD default if the fetch fails.

    Returns rate as a percentage (e.g. 3.6 means 3.6%).
    """
    currency = currency.upper()

    if currency == "USD":
        try:
            ticker = yf.Ticker("^IRX")
            hist = ticker.history(period="5d")
            if not hist.empty:
                return round(float(hist["Close"].iloc[-1]), 2)
        except Exception:
            pass

    return _DEFAULT_RATES.get(currency, _DEFAULT_RATES["USD"])


def _d1_d2(S, K, T, r, sigma):
    """Compute d1 and d2 for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type, ratio=1.0):
    """Black-Scholes option price adjusted for warrant ratio.

    Parameters
    ----------
    S : float – underlying price
    K : float – strike price (same currency as S)
    T : float – time to expiry in years
    r : float – risk-free rate as decimal (e.g. 0.02)
    sigma : float – volatility as decimal (e.g. 0.30)
    option_type : str – "call" or "put"
    ratio : float – fraction of underlying per warrant (e.g. 0.1 means 10 warrants = 1 share)
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price * ratio


def implied_volatility(market_price, S, K, T, r, option_type, ratio=1.0):
    """Solve for volatility that makes bs_price equal the market price.

    Uses Brent's method on the bracket [0.0001, 5.0].
    Raises ValueError if no solution exists.
    """
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type, ratio) - market_price

    try:
        return brentq(objective, 0.0001, 5.0, xtol=1e-6)
    except ValueError:
        raise ValueError(
            "Could not calculate implied volatility — "
            "check that the warrant price is consistent with the inputs."
        )


def greeks(S, K, T, r, sigma, option_type, ratio=1.0):
    """Compute Greeks for a warrant.

    Delta is the raw Black-Scholes delta (not ratio-adjusted), as is standard.
    Other Greeks are ratio-adjusted to reflect per-warrant sensitivity.

    ratio : float – fraction of underlying per warrant (e.g. 0.1 means 10 warrants = 1 share)

    Returns dict with delta (raw), gamma, theta (per day), vega (per 1% vol),
    rho (per 1% rate).
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    exp_rT = math.exp(-r * T)

    # Raw Greeks (one vanilla option)
    if option_type == "call":
        raw_delta = norm.cdf(d1)
        raw_rho = K * T * exp_rT * norm.cdf(d2)
    else:
        raw_delta = norm.cdf(d1) - 1
        raw_rho = -K * T * exp_rT * norm.cdf(-d2)

    raw_gamma = pdf_d1 / (S * sigma * sqrt_T)
    shared_term = -(S * pdf_d1 * sigma) / (2 * sqrt_T)
    if option_type == "call":
        raw_theta = shared_term - r * K * exp_rT * norm.cdf(d2)
    else:
        raw_theta = shared_term + r * K * exp_rT * norm.cdf(-d2)

    raw_vega = S * pdf_d1 * sqrt_T

    return {
        "delta": raw_delta,  # raw BS delta, not ratio-adjusted
        "gamma": raw_gamma,  # raw BS gamma (derivative of delta), not ratio-adjusted
        "theta": (raw_theta / 365) * ratio,
        "vega": (raw_vega / 100) * ratio,
        "rho": (raw_rho / 100) * ratio,
    }


def scenario_price(S_exit, T_exit, r, sigma, K, option_type, ratio=1.0):
    """Project warrant price at a given exit underlying price and time.

    Parameters
    ----------
    S_exit : float – projected underlying price at exit
    T_exit : float – remaining time to expiry at exit (years)
    r, sigma, K, option_type, ratio : same as bs_price
    """
    if T_exit <= 0:
        # At expiry, return intrinsic value
        if option_type == "call":
            return max(S_exit - K, 0) * ratio
        else:
            return max(K - S_exit, 0) * ratio
    return bs_price(S_exit, K, T_exit, r, sigma, option_type, ratio)
