from __future__ import annotations

import math
import statistics


def calculate_valuation(data: dict, custom_eps: float | None = None) -> dict:
    """Calculate fair value, entry price, and exit price.

    Blends PEG-implied fair value (60%) with growth-adjusted historical percentile (40%).
    """
    eps = custom_eps if custom_eps is not None else data.get("trailing_eps", 0)
    growth_5y = data.get("growth_5y", 0)
    current_price = data.get("current_price", 0)
    beta = data.get("beta", 1.0) or 1.0

    # Compute blended growth for PEG-implied valuation (same logic as scorer)
    blended_growth = _compute_blended_growth(data)

    # Method 1: PEG-Implied Fair Value (uses blended growth)
    peg_result = _peg_implied_fair_value(eps, blended_growth or growth_5y)

    # Method 2: Growth-Adjusted Historical Percentile
    hist_result = _historical_adjusted_fair_value(data, eps, growth_5y)

    # Blend
    peg_price = peg_result["fair_price"]
    hist_price = hist_result["fair_price"]

    if peg_price is not None and hist_price is not None:
        fair_value = peg_price * 0.6 + hist_price * 0.4
    elif peg_price is not None:
        fair_value = peg_price
    elif hist_price is not None:
        fair_value = hist_price
    else:
        fair_value = None

    # Entry with beta-adjusted margin of safety, Exit with beta-adjusted premium
    margin_of_safety = _calculate_margin(beta)
    exit_premium = _calculate_exit_premium(beta)

    entry_price = None
    exit_price = None
    if fair_value is not None:
        entry_price = round(fair_value * (1 - margin_of_safety), 2)
        exit_price = round(fair_value * (1 + exit_premium), 2)
        fair_value = round(fair_value, 2)

    return {
        "fair_value": fair_value,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "margin_of_safety": margin_of_safety,
        "exit_premium": exit_premium,
        "peg_method": peg_result,
        "historical_method": hist_result,
    }


def _dampen_growth(growth: float, k: float = 15.0) -> float:
    """Log-dampen a growth rate to compress high values."""
    if growth <= 0:
        return growth
    return k * math.log(1 + growth / k)


def _compute_blended_growth(data: dict) -> float | None:
    """Compute 3-horizon blended growth: 0y (35%) + +1y (35%) + 5y dampened (30%)."""
    growth_0y = data.get("growth_current_year")
    growth_1y = data.get("growth_next_year")
    growth_5y = data.get("growth_5y")

    components = []
    if growth_0y and growth_0y > 0:
        components.append((growth_0y, 0.35))
    if growth_1y and growth_1y > 0:
        components.append((growth_1y, 0.35))
    if growth_5y and growth_5y > 0:
        components.append((_dampen_growth(growth_5y), 0.30))

    if not components:
        return None

    total_weight = sum(w for _, w in components)
    return sum(val * (w / total_weight) for val, w in components)


def compute_historical_pe_series(data: dict) -> list[dict]:
    """Compute dated historical P/E series for charting.

    Returns list of {"date": datetime, "pe": float} dicts.
    Uses time-appropriate EPS (same logic as valuation).
    """
    historical_prices = data.get("historical_prices")
    annual_eps = data.get("annual_eps_history")
    current_eps = data.get("trailing_eps", 0)

    if not historical_prices or not current_eps or current_eps <= 0:
        return []

    # Detect currency/ADR mismatch
    use_current_only = False
    if annual_eps:
        latest_annual_eps = annual_eps[-1]["eps"]
        ratio = latest_annual_eps / current_eps if current_eps > 0 else 1.0
        if ratio > 3.0 or ratio < 0.33:
            use_current_only = True
    else:
        use_current_only = True

    series = []
    for point in historical_prices:
        price = point["price"]
        if price <= 0:
            continue

        if use_current_only:
            pe = price / current_eps
        else:
            price_date = point["date"].replace(tzinfo=None)
            applicable_eps = None
            for eps_record in annual_eps:
                if eps_record["date"].replace(tzinfo=None) <= price_date:
                    applicable_eps = eps_record["eps"]
            if not applicable_eps or applicable_eps <= 0:
                continue
            pe = price / applicable_eps

        if 0 < pe < 200:
            series.append({"date": point["date"], "pe": round(pe, 2)})

    return series


def _peg_implied_fair_value(eps: float, growth: float) -> dict:
    """PEG=1 implies Fair P/E = Growth Rate. Fair Price = Fair P/E * EPS."""
    if not eps or eps <= 0 or not growth or growth <= 0:
        return {"fair_pe": None, "fair_price": None}

    fair_pe = max(growth, 8.0)  # Floor at P/E of 8
    fair_price = round(fair_pe * eps, 2)

    return {"fair_pe": round(fair_pe, 2), "fair_price": fair_price}


def _historical_adjusted_fair_value(data: dict, eps: float, current_growth: float) -> dict:
    """Compute fair value from growth-adjusted historical P/E median.

    Uses time-appropriate EPS for each historical price point (from annual
    income statements) rather than dividing all prices by current EPS.
    """
    historical_prices = data.get("historical_prices")
    annual_eps = data.get("annual_eps_history")
    historical_growth = data.get("historical_growth_5y")

    if not historical_prices or not eps or eps <= 0:
        return {"median_pe": None, "adjusted_pe": None, "fair_price": None, "growth_ratio": None}

    # Compute historical P/E using the EPS that was current at each price point
    historical_pes = _compute_historical_pes(historical_prices, annual_eps, eps)
    historical_pes = [pe for pe in historical_pes if 0 < pe < 200]  # Filter outliers

    if not historical_pes:
        return {"median_pe": None, "adjusted_pe": None, "fair_price": None, "growth_ratio": None}

    median_pe = statistics.median(historical_pes)

    # Growth adjustment
    if historical_growth and historical_growth > 0 and current_growth and current_growth > 0:
        growth_ratio = max(current_growth / historical_growth, 0.3)  # Floor at 0.3
    else:
        growth_ratio = 1.0  # No adjustment if historical growth unavailable

    adjusted_pe = median_pe * growth_ratio
    fair_price = round(adjusted_pe * eps, 2)

    return {
        "median_pe": round(median_pe, 2),
        "adjusted_pe": round(adjusted_pe, 2),
        "fair_price": fair_price,
        "growth_ratio": round(growth_ratio, 2),
    }


def _compute_historical_pes(
    prices: list[dict], annual_eps: list[dict] | None, current_eps: float
) -> list[float]:
    """Compute P/E for each historical price using the EPS current at that time.

    For each monthly price, finds the most recent annual EPS that was reported
    before that date. Falls back to current EPS for dates without coverage.

    Detects currency/ADR mismatches (e.g., TSM reports EPS in TWD but trades
    in USD) by comparing annual EPS to the info-provided trailing EPS. If they
    differ by more than 3x, the annual EPS is in a different currency and we
    fall back to current (USD-adjusted) EPS for all price points.
    """
    if not annual_eps:
        return [p["price"] / current_eps for p in prices if p["price"] > 0]

    # Detect currency mismatch: compare latest annual EPS to trailing EPS from info
    latest_annual_eps = annual_eps[-1]["eps"]
    ratio = latest_annual_eps / current_eps if current_eps > 0 else 1.0
    if ratio > 3.0 or ratio < 0.33:
        # Currency or ADR mismatch — annual EPS is in a different unit than prices
        return [p["price"] / current_eps for p in prices if p["price"] > 0]

    pes = []
    for point in prices:
        price = point["price"]
        if price <= 0:
            continue

        price_date = point["date"].replace(tzinfo=None)

        # Find the most recent annual EPS reported before this price date
        applicable_eps = None
        for eps_record in annual_eps:
            if eps_record["date"].replace(tzinfo=None) <= price_date:
                applicable_eps = eps_record["eps"]

        if applicable_eps and applicable_eps > 0:
            pes.append(price / applicable_eps)
        # Skip price points where we have no applicable EPS (before earliest report)

    return pes


def _calculate_margin(beta: float) -> float:
    """Beta-adjusted margin of safety. Base 5%, clamped to [0%, 10%]."""
    margin = 0.05 + (beta - 1.0) * 0.05
    return max(0.00, min(0.10, margin))


def _calculate_exit_premium(beta: float) -> float:
    """Beta-adjusted exit premium. Base 30%, clamped to [20%, 40%]."""
    premium = 0.30 + (beta - 1.0) * 0.10
    return max(0.20, min(0.40, premium))
