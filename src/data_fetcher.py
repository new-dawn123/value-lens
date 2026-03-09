from __future__ import annotations

import logging
import time
import yfinance as yf
import pandas as pd

# Suppress noisy yfinance HTTP error logs
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds: 2, 4, 8


def _fetch_finviz_fundamentals(ticker: str) -> dict | None:
    """Fetch fundamental snapshot from finviz. Returns dict or None on failure."""
    try:
        from finvizfinance.quote import finvizfinance
        stock = finvizfinance(ticker)
        return stock.ticker_fundament()
    except Exception:
        return None


def _parse_finviz_pct(value: str) -> float | None:
    """Parse a finviz percentage string like '10.20%' or '-3.50%' to a float."""
    if not value or value == "-":
        return None
    try:
        return float(value.replace("%", ""))
    except (ValueError, TypeError):
        return None


def _parse_finviz_dual_pct(value: str, index: int = 1) -> float | None:
    """Parse a finviz dual-value field like '6.89% 17.91%'.

    index=0 → first value (3Y), index=1 → second value (5Y).
    """
    if not value or value == "-":
        return None
    try:
        parts = value.split()
        if len(parts) > index:
            return float(parts[index].replace("%", ""))
    except (ValueError, TypeError, IndexError):
        pass
    return None


def fetch_stock_data(ticker: str) -> dict:
    """Fetch all fundamental data needed for analysis from yfinance + finviz."""
    stock, info = _fetch_info_with_retry(ticker)

    if not info or info.get("quoteType") is None:
        raise ValueError(f"Ticker '{ticker}' not found or no data available")

    # Fetch finviz fundamentals (single call, used for growth fields)
    finviz = _fetch_finviz_fundamentals(ticker)

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

    # 5-year estimated EPS growth (finviz primary, yfinance fallback)
    growth_5y, growth_5y_source = _get_growth_estimate(stock, finviz)
    data["growth_5y"] = growth_5y
    data["growth_5y_source"] = growth_5y_source

    # Short-term EPS growth (0y and +1y) — finviz primary, yfinance fallback
    short_term = _get_short_term_growth(stock, finviz)
    data["growth_current_year"] = short_term["growth_current_year"]
    data["growth_current_year_source"] = short_term["growth_current_year_source"]
    data["growth_next_year"] = short_term["growth_next_year"]
    data["growth_next_year_source"] = short_term["growth_next_year_source"]

    # Next-year revenue growth
    data["revenue_growth_next_year"] = _get_revenue_growth(stock)

    # EPS revisions
    data["eps_revisions"] = _get_eps_revisions(stock)

    # Earnings history (last 4 quarters)
    data["earnings_history"] = _get_earnings_history(stock)

    # Historical 5Y growth (past) — finviz primary, yfinance fallback
    hist_growth, hist_growth_source = _get_historical_growth(stock, finviz)
    data["historical_growth_5y"] = hist_growth
    data["historical_growth_5y_source"] = hist_growth_source

    # Historical 3Y EPS growth (finviz only)
    data["historical_growth_3y"] = _parse_finviz_dual_pct(
        finviz.get("EPS past 3/5Y"), index=0,
    ) if finviz else None

    # Sales growth past 3Y and 5Y (finviz only)
    data["sales_growth_3y"] = _parse_finviz_dual_pct(
        finviz.get("Sales past 3/5Y"), index=0,
    ) if finviz else None
    data["sales_growth_5y"] = _parse_finviz_dual_pct(
        finviz.get("Sales past 3/5Y"), index=1,
    ) if finviz else None

    # 5-year historical prices for historical P/E computation
    data["historical_prices"] = _get_historical_prices(stock)

    # Annual EPS history for time-accurate historical P/E
    data["annual_eps_history"] = _get_annual_eps_history(stock)

    # Quarterly EPS history for historical forward P/E
    data["quarterly_eps_history"] = _get_quarterly_eps_history(stock)

    # Currency conversion: convert income-statement EPS to price currency
    price_currency = info.get("currency")
    financial_currency = info.get("financialCurrency")
    data["currency"] = price_currency
    data["financial_currency"] = financial_currency
    data["fx_converted"] = False

    if (
        price_currency
        and financial_currency
        and price_currency != financial_currency
    ):
        fx_rate = _fetch_fx_rate(financial_currency, price_currency)
        if fx_rate is not None:
            _convert_eps_history(data["annual_eps_history"], fx_rate)
            _convert_eps_history(data["quarterly_eps_history"], fx_rate)
            # P/S ratio: Yahoo computes MarketCap / Revenue with mixed currencies
            # for ADRs/foreign listings. Divide by fx_rate to normalize.
            if data["ps_ratio"] is not None:
                data["ps_ratio"] = data["ps_ratio"] / fx_rate
            data["fx_converted"] = True
            data["fx_rate"] = fx_rate

    # Legacy flag: true when EPS is still in the wrong currency (FX fetch failed)
    data["currency_mismatch"] = _detect_currency_mismatch(data)

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


def _get_growth_estimate(
    stock: yf.Ticker, finviz: dict | None = None,
) -> tuple[float | None, str]:
    """Extract estimated EPS growth rate with fallback chain.

    Priority:
    1. Finviz 'EPS next 5Y'
    2. Long-term growth (LTG) from yfinance growth_estimates (5Y estimate)
    3. Average of current-year and next-year growth from yfinance earnings_estimate
    4. earningsGrowth from yfinance ticker info

    Returns (value, source_label).
    """
    # Try 1: Finviz 'EPS next 5Y'
    if finviz:
        val = _parse_finviz_pct(finviz.get("EPS next 5Y"))
        if val is not None:
            return val, "finviz"

    # Try 2: LTG from growth_estimates
    try:
        ge = stock.growth_estimates
        if ge is not None and not ge.empty:
            for row_label in ["LTG", "+5y"]:
                if row_label in ge.index:
                    for col in ["stockTrend", "stock"]:
                        if col in ge.columns:
                            val = ge.loc[row_label, col]
                            if val is not None and pd.notna(val):
                                return float(val) * 100, "yfinance"
    except Exception:
        pass

    # Try 3: Average of 0y and +1y growth from earnings_estimate
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
                return (sum(growth_values) / len(growth_values)) * 100, "yfinance"
    except Exception:
        pass

    # Try 4: earningsGrowth from info
    try:
        info = stock.info
        eg = info.get("earningsGrowth")
        if eg is not None:
            return float(eg) * 100, "yfinance"
    except Exception:
        pass

    return None, "N/A"


def _get_short_term_growth(
    stock: yf.Ticker, finviz: dict | None = None,
) -> dict:
    """Extract current-year (0y) and next-year (+1y) EPS growth.

    Finviz primary, yfinance fallback.  Used for blended dampened PEG.
    Returns dict with growth values and their sources.
    """
    result = {
        "growth_current_year": None, "growth_current_year_source": "N/A",
        "growth_next_year": None, "growth_next_year_source": "N/A",
    }

    # Try finviz first
    if finviz:
        val = _parse_finviz_pct(finviz.get("EPS this Y"))
        if val is not None:
            result["growth_current_year"] = val
            result["growth_current_year_source"] = "finviz"
        # EPS next Y: finviz has both an absolute and a percentage field
        val = _parse_finviz_pct(finviz.get("EPS next Y Percentage"))
        if val is not None:
            result["growth_next_year"] = val
            result["growth_next_year_source"] = "finviz"

    # Fallback to yfinance for any still-missing fields
    try:
        ee = stock.earnings_estimate
        if ee is not None and not ee.empty and "growth" in ee.columns:
            for period, key in [("0y", "growth_current_year"), ("+1y", "growth_next_year")]:
                if result[key] is None and period in ee.index:
                    val = ee.loc[period, "growth"]
                    if val is not None and pd.notna(val):
                        result[key] = float(val) * 100
                        result[f"{key}_source"] = "yfinance"
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


def _get_historical_growth(
    stock: yf.Ticker, finviz: dict | None = None,
) -> tuple[float | None, str]:
    """Extract historical EPS growth rate.

    Priority:
    1. Finviz 'EPS past 5Y'
    2. -5y row from yfinance growth_estimates
    3. Log-linear regression on annual Diluted EPS from yfinance income statement.

    Returns (value, source_label).
    """
    # Try 1: Finviz 'EPS past 3/5Y' — second value is the 5Y growth
    if finviz:
        val = _parse_finviz_dual_pct(finviz.get("EPS past 3/5Y"), index=1)
        if val is not None:
            return val, "finviz"

    # Try 2: -5y from growth_estimates
    try:
        ge = stock.growth_estimates
        if ge is not None and not ge.empty and "-5y" in ge.index:
            for col in ["stockTrend", "stock"]:
                if col in ge.columns:
                    val = ge.loc["-5y", col]
                    if val is not None and pd.notna(val):
                        return float(val) * 100, "yfinance"
    except Exception:
        pass

    # Try 3: Log-linear regression on annual Diluted EPS
    try:
        import numpy as np

        inc = stock.income_stmt
        if inc is not None and not inc.empty and "Diluted EPS" in inc.index:
            eps_row = inc.loc["Diluted EPS"]
            points = []
            for date, val in eps_row.items():
                if val is not None and pd.notna(val) and float(val) > 0:
                    points.append((date, float(val)))
            points.sort(key=lambda x: x[0])

            if len(points) >= 2:
                span = (points[-1][0] - points[0][0]).days / 365.25
                if span >= 0.99:
                    years = np.array(
                        [(d - points[0][0]).days / 365.25 for d, _ in points]
                    )
                    log_eps = np.log(np.array([e for _, e in points]))
                    slope, _ = np.polyfit(years, log_eps, 1)
                    cagr = (np.exp(slope) - 1) * 100
                    return round(float(cagr), 2), "yfinance (CAGR)"
    except Exception:
        pass

    return None, "N/A"


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


def _fetch_fx_rate(from_currency: str, to_currency: str) -> float | None:
    """Fetch current FX rate from yfinance. Returns rate to multiply by."""
    try:
        pair = f"{from_currency}{to_currency}=X"
        info = yf.Ticker(pair).info
        rate = info.get("regularMarketPrice") or info.get("previousClose")
        if rate and rate > 0:
            return float(rate)
    except Exception:
        pass
    return None


def _convert_eps_history(eps_history: list[dict] | None, fx_rate: float) -> None:
    """Convert EPS values in-place by multiplying with fx_rate."""
    if not eps_history:
        return
    for record in eps_history:
        record["eps"] = record["eps"] * fx_rate


def _detect_currency_mismatch(data: dict) -> bool:
    """Detect if annual EPS is still in a different currency than trailing EPS.

    This is a safety-net check that catches cases where FX conversion failed
    or was not attempted. Uses a 3x divergence threshold.
    """
    annual_eps = data.get("annual_eps_history")
    trailing_eps = data.get("trailing_eps")
    if not annual_eps or not trailing_eps or trailing_eps <= 0:
        return False
    ratio = annual_eps[-1]["eps"] / trailing_eps
    return ratio > 3.0 or ratio < 0.33


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
