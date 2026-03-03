from __future__ import annotations

import logging
import time
import yfinance as yf
import pandas as pd

# Suppress noisy yfinance HTTP error logs
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds: 2, 4, 8


def fetch_stock_data(ticker: str) -> dict:
    """Fetch all fundamental data needed for analysis from yfinance."""
    stock, info = _fetch_info_with_retry(ticker)

    if not info or info.get("quoteType") is None:
        raise ValueError(f"Ticker '{ticker}' not found or no data available")

    data = {
        "name": info.get("shortName") or info.get("longName") or ticker,
        "website": info.get("website"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "trailing_eps": info.get("trailingEps"),
        "ps_ratio": info.get("priceToSalesTrailing12Months") or info.get("priceToSalesTrailing12M"),
        "beta": info.get("beta"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "market_cap": info.get("marketCap"),
    }

    # 5-year estimated EPS growth
    data["growth_5y"] = _get_growth_estimate(stock)

    # Short-term EPS growth (0y and +1y) for blended PEG
    short_term = _get_short_term_growth(stock)
    data["growth_current_year"] = short_term["growth_current_year"]
    data["growth_next_year"] = short_term["growth_next_year"]

    # Next-year revenue growth
    data["revenue_growth_next_year"] = _get_revenue_growth(stock)

    # EPS revisions
    data["eps_revisions"] = _get_eps_revisions(stock)

    # Earnings history (last 4 quarters)
    data["earnings_history"] = _get_earnings_history(stock)

    # Historical 5Y growth (past)
    data["historical_growth_5y"] = _get_historical_growth(stock)

    # 5-year historical prices for historical P/E computation
    data["historical_prices"] = _get_historical_prices(stock)

    # Annual EPS history for time-accurate historical P/E
    data["annual_eps_history"] = _get_annual_eps_history(stock)

    # Quarterly EPS history for historical forward P/E
    data["quarterly_eps_history"] = _get_quarterly_eps_history(stock)

    return data


def _fetch_info_with_retry(ticker: str) -> tuple[yf.Ticker, dict]:
    """Fetch ticker info with exponential backoff on rate limits."""
    for attempt in range(_MAX_RETRIES):
        stock = yf.Ticker(ticker)
        info = stock.info
        # yfinance returns a near-empty dict on rate limit (no quoteType, no shortName)
        if info and info.get("quoteType") is not None:
            return stock, info
        if attempt < _MAX_RETRIES - 1:
            wait = _BACKOFF_BASE ** (attempt + 1)
            time.sleep(wait)
    return stock, info


def _get_growth_estimate(stock: yf.Ticker) -> float | None:
    """Extract estimated EPS growth rate with fallback chain.

    Priority:
    1. Long-term growth (LTG) from growth_estimates (5Y estimate)
    2. Average of current-year and next-year growth from earnings_estimate
    3. earningsGrowth from ticker info
    """
    # Try 1: LTG from growth_estimates
    try:
        ge = stock.growth_estimates
        if ge is not None and not ge.empty:
            # Check for LTG (long-term growth) or +5y row
            for row_label in ["LTG", "+5y"]:
                if row_label in ge.index:
                    for col in ["stockTrend", "stock"]:
                        if col in ge.columns:
                            val = ge.loc[row_label, col]
                            if val is not None and pd.notna(val):
                                return float(val) * 100
    except Exception:
        pass

    # Try 2: Average of 0y and +1y growth from earnings_estimate
    try:
        ee = stock.earnings_estimate
        if ee is not None and not ee.empty and "growth" in ee.columns:
            growth_values = []
            for period in ["0y", "+1y"]:
                if period in ee.index:
                    val = ee.loc[period, "growth"]
                    if val is not None and pd.notna(val):
                        growth_values.append(float(val))
            if growth_values:
                return (sum(growth_values) / len(growth_values)) * 100
    except Exception:
        pass

    # Try 3: earningsGrowth from info
    try:
        info = stock.info
        eg = info.get("earningsGrowth")
        if eg is not None:
            return float(eg) * 100
    except Exception:
        pass

    return None


def _get_short_term_growth(stock: yf.Ticker) -> dict:
    """Extract current-year (0y) and next-year (+1y) EPS growth from earnings_estimate.

    These are used for the blended dampened PEG calculation.
    Returns a dict with growth_current_year and growth_next_year (both as percentages or None).
    """
    result = {"growth_current_year": None, "growth_next_year": None}
    try:
        ee = stock.earnings_estimate
        if ee is not None and not ee.empty and "growth" in ee.columns:
            for period, key in [("0y", "growth_current_year"), ("+1y", "growth_next_year")]:
                if period in ee.index:
                    val = ee.loc[period, "growth"]
                    if val is not None and pd.notna(val):
                        result[key] = float(val) * 100
    except Exception:
        pass
    return result


def _get_revenue_growth(stock: yf.Ticker) -> float | None:
    """Extract next-year estimated revenue growth."""
    try:
        re = stock.revenue_estimate
        if re is not None and not re.empty and "+1y" in re.index:
            val = re.loc["+1y", "growth"] if "growth" in re.columns else None
            if val is not None and pd.notna(val):
                return float(val) * 100
    except Exception:
        pass
    return None


def _get_eps_revisions(stock: yf.Ticker) -> dict | None:
    """Extract EPS revision data (up/down counts for 7d and 30d)."""
    try:
        er = stock.eps_revisions
        if er is not None and not er.empty:
            revisions = {}
            for period in ["0q", "+1q", "0y", "+1y"]:
                if period in er.index:
                    row = er.loc[period]
                    revisions[period] = {
                        "up_7d": _safe_int(row.get("upLast7days")),
                        "up_30d": _safe_int(row.get("upLast30days")),
                        "down_7d": _safe_int(row.get("downLast7days")),
                        "down_30d": _safe_int(row.get("downLast30days")),
                    }
            return revisions if revisions else None
    except Exception:
        pass
    return None


def _get_earnings_history(stock: yf.Ticker) -> list[dict] | None:
    """Extract last 4 quarters of earnings history."""
    try:
        eh = stock.earnings_history
        if eh is not None and not eh.empty:
            records = []
            for _, row in eh.iterrows():
                records.append({
                    "eps_estimate": _safe_float(row.get("epsEstimate")),
                    "eps_actual": _safe_float(row.get("epsActual")),
                    "surprise_pct": _safe_float(row.get("surprisePercent")),
                })
            return records[-4:] if records else None
    except Exception:
        pass
    return None


def _get_historical_growth(stock: yf.Ticker) -> float | None:
    """Extract historical EPS growth rate.

    Priority:
    1. -5y row from growth_estimates (if Yahoo still provides it)
    2. CAGR computed from annual income statement Diluted EPS
    """
    # Try 1: -5y from growth_estimates
    try:
        ge = stock.growth_estimates
        if ge is not None and not ge.empty and "-5y" in ge.index:
            for col in ["stockTrend", "stock"]:
                if col in ge.columns:
                    val = ge.loc["-5y", col]
                    if val is not None and pd.notna(val):
                        return float(val) * 100
    except Exception:
        pass

    # Try 2: CAGR from annual Diluted EPS (income statement)
    try:
        inc = stock.income_stmt
        if inc is not None and not inc.empty and "Diluted EPS" in inc.index:
            eps_row = inc.loc["Diluted EPS"]
            # Collect valid positive EPS with dates, sorted oldest first
            points = []
            for date, val in eps_row.items():
                if val is not None and pd.notna(val) and float(val) > 0:
                    points.append((date, float(val)))
            points.sort(key=lambda x: x[0])

            if len(points) >= 2:
                oldest_date, oldest_eps = points[0]
                newest_date, newest_eps = points[-1]
                years = (newest_date - oldest_date).days / 365.25
                if years >= 1.0 and oldest_eps > 0:
                    cagr = ((newest_eps / oldest_eps) ** (1.0 / years) - 1) * 100
                    return round(cagr, 2)
    except Exception:
        pass

    return None


def _get_historical_prices(stock: yf.Ticker) -> list[dict] | None:
    """Get monthly closing prices with dates for the last 5 years."""
    try:
        hist = stock.history(period="5y", interval="1mo")
        if hist is not None and not hist.empty:
            records = []
            for date, row in hist.iterrows():
                price = row.get("Close")
                if price is not None and pd.notna(price) and price > 0:
                    records.append({"date": date.to_pydatetime(), "price": float(price)})
            return records if records else None
    except Exception:
        pass
    return None


def _get_annual_eps_history(stock: yf.Ticker) -> list[dict] | None:
    """Get annual Diluted EPS from income statement for historical P/E computation."""
    try:
        inc = stock.income_stmt
        if inc is not None and not inc.empty and "Diluted EPS" in inc.index:
            eps_row = inc.loc["Diluted EPS"]
            records = []
            for date, eps_val in eps_row.items():
                if eps_val is not None and pd.notna(eps_val) and float(eps_val) > 0:
                    records.append({"date": date.to_pydatetime(), "eps": float(eps_val)})
            # Sort oldest first
            records.sort(key=lambda r: r["date"])
            return records if records else None
    except Exception:
        pass
    return None


def _get_quarterly_eps_history(stock: yf.Ticker) -> list[dict] | None:
    """Get quarterly Diluted EPS from quarterly income statement.

    Returns list of {"date": datetime, "eps": float} sorted oldest-first.
    Unlike the annual version, individual quarters may have negative EPS
    (only the 4-quarter sum matters for forward P/E).
    """
    try:
        inc = stock.quarterly_income_stmt
        if inc is not None and not inc.empty and "Diluted EPS" in inc.index:
            eps_row = inc.loc["Diluted EPS"]
            records = []
            for date, eps_val in eps_row.items():
                if eps_val is not None and pd.notna(eps_val):
                    records.append({"date": date.to_pydatetime(), "eps": float(eps_val)})
            records.sort(key=lambda r: r["date"])
            return records if records else None
    except Exception:
        pass
    return None


def _safe_float(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0
