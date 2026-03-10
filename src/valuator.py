from __future__ import annotations

import math
import statistics

from src.scorer import BASE_PE, _GROWTH_DAMPEN_K, _fair_pe

def calculate_valuation(
    data: dict,
    custom_eps: float | None = None,
    custom_growth: float | None = None,
    scores: dict | None = None,
    disregard_hist_premium: bool = False,
) -> dict:
    """Calculate fair value, entry price, and exit price.

    Single-flow approach:
        fair_value = PEG_fair_price × historical_premium

    1. PEG fair price: BASE_PE × (1 + dampened_growth/100)^5 × EPS
    2. Historical premium: median_actual_PE / model_fair_PE(raw_historical_growth)
       — captures persistent market premium/discount (moat, quality, etc.)
       — clamped [0.80, 1.20], defaults to 1.0 when data is insufficient.

    Entry price: beta-scaled growth scenario (pessimistic fair value).
    Exit price: historical P/E stretch premium above entry.

    Custom growth is dampened the same way as fetched growth.
    """
    eps = custom_eps if custom_eps is not None else data.get("trailing_eps", 0)
    beta = data.get("beta", 1.0) or 1.0

    # Growth: use custom override (dampened) or compute dampened 5Y
    if custom_growth is not None:
        growth_for_peg = _dampen_growth(custom_growth) if custom_growth > 0 else custom_growth
    else:
        growth_for_peg = _compute_effective_growth(data)

    # Historical P/E series (used by both premium and exit calculations)
    hist_pes = _get_historical_pes(data)

    # Step 1: PEG-implied fair value (core model)
    peg_result = _peg_implied_fair_value(eps, growth_for_peg)

    # Step 2: Historical premium/discount multiplier
    hist_premium = _compute_historical_premium(data, hist_pes)

    # Effective multiplier (1.0 when disregarded)
    eff_premium = 1.0 if disregard_hist_premium else hist_premium["premium"]

    # Combine: fair_value = peg_price × premium
    fair_value = peg_result["fair_price"]
    if fair_value is not None:
        fair_value = fair_value * eff_premium

    # Entry: beta-scaled growth scenario
    entry_price = None
    margin_of_safety = 0.0
    if fair_value is not None and growth_for_peg is not None:
        entry_price, margin_of_safety = _calculate_entry(
            fair_value, growth_for_peg, eps, beta,
            eff_premium,
        )

    # Exit: historical P/E stretch above entry
    exit_price = None
    exit_premium = 0.0
    pe_stretch = 1.0
    if entry_price is not None:
        exit_price, exit_premium, pe_stretch = _calculate_exit(
            entry_price, hist_pes,
        )

    if fair_value is not None:
        fair_value = round(fair_value, 2)

    return {
        "fair_value": fair_value,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "margin_of_safety": margin_of_safety,
        "exit_premium": exit_premium,
        "pe_stretch": pe_stretch,
        "peg_method": peg_result,
        "historical_premium": hist_premium,
    }


def _dampen_growth(growth: float, k: float = _GROWTH_DAMPEN_K) -> float:
    """Log-dampen a positive growth rate to compress optimistic estimates."""
    if growth <= 0:
        return growth
    return k * math.log(1 + growth / k)


def _compute_effective_growth(data: dict) -> float | None:
    """Compute effective growth: dampened 5Y estimate (supports negative)."""
    growth_5y = data.get("growth_5y")

    if growth_5y is None:
        return None

    if growth_5y > 0:
        return _dampen_growth(growth_5y)

    return growth_5y


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


def compute_historical_forward_pe_series(data: dict) -> list[dict]:
    """Compute historical forward P/E series for charting.

    Combines two data sources:

    1. **Annual**: next fiscal year's actual EPS. Valid when the FY end
       is within ~400 days (13 months) so the EPS roughly represents
       the next 12 months.
    2. **Quarterly**: actual quarterly EPS from yfinance (~5 records).
       Detects missing quarters by gap to first available record.
       effective_N = available records minus missing front quarters.

    Blending logic per price point:
        effective_N >= 4: quarterly only (full 12-month rolling view)
        effective_N 1-3:  annual*(4-N)/4 + sum(N quarters)
        effective_N 0:    annual only
        neither valid:    skip (chart stops)

    The chart stops when quarterly drops below 4 and there is no
    valid annual EPS to blend with (i.e., the last annual record
    is the current fiscal year with no future FY available).

    Returns list of {"date": datetime, "pe": float} dicts.
    """
    historical_prices = data.get("historical_prices")
    annual_eps = data.get("annual_eps_history")
    quarterly_eps = data.get("quarterly_eps_history")
    current_eps = data.get("trailing_eps", 0)

    if not historical_prices or not annual_eps or len(annual_eps) < 2:
        return []

    # Detect currency/ADR mismatch (same logic as trailing P/E)
    latest_annual_eps = annual_eps[-1]["eps"]
    if current_eps and current_eps > 0:
        ratio = latest_annual_eps / current_eps
        if ratio > 3.0 or ratio < 0.33:
            return []  # Currency mismatch
    elif not current_eps or current_eps <= 0:
        return []

    last_annual_idx = len(annual_eps) - 1

    series = []
    for point in historical_prices:
        price = point["price"]
        if price <= 0:
            continue

        price_date = point["date"].replace(tzinfo=None)

        # --- Annual: interpolate between next two FY EPS values ---
        # Weight by how much of each FY falls within the next 12 months.
        # e.g. 200 days to FY1 end → 200/365 of FY1 + 165/365 of FY2.
        # Smooths the staircase that occurs at fiscal year boundaries.
        annual_fwd_eps = None
        for i, eps_record in enumerate(annual_eps):
            fy_end = eps_record["date"].replace(tzinfo=None)
            if fy_end > price_date:
                days_to_fy_end = (fy_end - price_date).days
                is_last = (i == last_annual_idx)
                if days_to_fy_end <= 400 and not is_last:
                    fy1_eps = eps_record["eps"]
                    fy2_eps = annual_eps[i + 1]["eps"]
                    if fy1_eps and fy1_eps > 0 and fy2_eps and fy2_eps > 0:
                        w1 = min(days_to_fy_end, 365) / 365
                        annual_fwd_eps = fy1_eps * w1 + fy2_eps * (1 - w1)
                break

        # --- Quarterly: detect missing front quarters via gap analysis ---
        effective_n = 0
        quarterly_sum = 0
        if quarterly_eps:
            candidates = [
                q for q in quarterly_eps
                if q["date"].replace(tzinfo=None) > price_date
            ]
            if candidates:
                gap_days = (candidates[0]["date"].replace(tzinfo=None) - price_date).days
                missing_front = max(0, (gap_days + 45) // 91 - 1)
                effective_n = min(4 - missing_front, len(candidates))
                effective_n = max(0, effective_n)
                quarterly_sum = sum(
                    q["eps"] for q in candidates[:effective_n]
                )

        # --- Combine ---
        if effective_n >= 4 and quarterly_sum > 0:
            # Full quarterly view — use as-is
            forward_eps = quarterly_sum
        elif effective_n > 0 and annual_fwd_eps:
            # Partial quarterly + annual fill for missing quarters
            forward_eps = annual_fwd_eps * (4 - effective_n) / 4 + quarterly_sum
        elif annual_fwd_eps:
            # Annual only (older dates before quarterly data)
            forward_eps = annual_fwd_eps
        else:
            continue

        if forward_eps > 0:
            pe = price / forward_eps
            if 0 < pe < 200:
                series.append({"date": point["date"], "pe": round(pe, 2)})

    return series


def _peg_implied_fair_value(eps: float, growth: float) -> dict:
    """5-year compounding fair value: Fair P/E = BASE_PE * (1 + g/100)^5.

    Pay today's no-growth multiple (12x) for the earnings the company
    will have in 5 years. Works for negative, zero, and positive growth.
    """
    if not eps or eps <= 0 or growth is None:
        return {"fair_pe": None, "fair_price": None}

    fair = _fair_pe(growth)
    if fair <= 0:
        return {"fair_pe": None, "fair_price": None}

    fair_price = round(fair * eps, 2)

    return {"fair_pe": round(fair, 2), "fair_price": fair_price}


def _compute_historical_premium(
    data: dict, hist_pes: list[float] | None = None,
) -> dict:
    """Compute a premium/discount multiplier from historical P/E vs model fair P/E.

    Compares how the market actually valued the stock (median trailing P/E over
    5 years) against what the compounding model says it *should* have traded at
    given its historical growth rate.

        model_fair_pe = BASE_PE × (1 + hist_growth/100)^5
        premium       = median_actual_pe / model_fair_pe   (clamped [0.85, 1.15])

    A premium > 1 means the market has historically paid more than the model
    predicts (brand moat, quality perception, etc.).  A discount < 1 means
    the market valued it below model expectations.

    Historical growth is NOT dampened — it is backward-looking fact, not an
    optimistic analyst estimate.

    Accepts an optional pre-computed hist_pes list to avoid recomputation
    (the same list is also used by the exit price calculation).

    Returns dict with premium, median_pe, model_pe, and detail fields.
    Returns premium=1.0 (neutral) when data is insufficient.
    """
    historical_growth = data.get("historical_growth_5y")

    neutral = {"premium": 1.0, "median_pe": None, "model_pe": None, "historical_growth": None}

    if hist_pes is None:
        hist_pes = _get_historical_pes(data)

    if not hist_pes:
        return neutral

    median_pe = statistics.median(hist_pes)

    # Need historical growth to compute model fair P/E for comparison
    if historical_growth is None:
        return {
            "premium": 1.0,
            "median_pe": round(median_pe, 2),
            "model_pe": None,
            "historical_growth": None,
        }

    # Model fair P/E for historical growth (raw, no dampening)
    model_pe = _fair_pe(historical_growth)

    if model_pe <= 0:
        return {
            "premium": 1.0,
            "median_pe": round(median_pe, 2),
            "model_pe": round(model_pe, 2),
            "historical_growth": round(historical_growth, 2),
        }

    # Premium: how much more/less the market paid vs model expectation
    premium = median_pe / model_pe
    premium = max(0.80, min(1.20, premium))  # Clamp ±20%

    return {
        "premium": round(premium, 2),
        "median_pe": round(median_pe, 2),
        "model_pe": round(model_pe, 2),
        "historical_growth": round(historical_growth, 2),
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


def _get_historical_pes(data: dict) -> list[float]:
    """Get filtered historical P/E values from stock data."""
    historical_prices = data.get("historical_prices")
    annual_eps = data.get("annual_eps_history")
    current_eps = data.get("trailing_eps", 0)

    if not historical_prices or not current_eps or current_eps <= 0:
        return []

    pes = _compute_historical_pes(historical_prices, annual_eps, current_eps)
    return [pe for pe in pes if 0 < pe < 200]


def _calculate_entry(
    fair_value: float,
    growth_for_peg: float,
    eps: float,
    beta: float,
    premium: float,
) -> tuple[float, float]:
    """Beta-scaled growth scenario entry price.

    Computes a pessimistic fair value by shocking growth downward based
    on beta.  Higher beta = larger shock = wider entry margin.

        shock = 0.10 × (0.5 + 0.5 × beta), clamped [0.05, 0.25]
        entry_growth = growth × (1 - shock)   [worse direction for neg growth]
        entry_price  = fair_pe(entry_growth) × EPS × premium
        entry discount clamped [3%, 15%]

    Returns (entry_price, entry_discount).
    """
    shock = 0.10 * (0.5 + 0.5 * beta)
    shock = max(0.05, min(0.25, shock))

    if growth_for_peg > 0:
        g_entry = growth_for_peg * (1 - shock)
    else:
        g_entry = growth_for_peg * (1 + shock)

    peg_entry = _peg_implied_fair_value(eps, g_entry)

    if peg_entry["fair_price"]:
        entry_raw = peg_entry["fair_price"] * premium
    else:
        entry_raw = fair_value * 0.95

    entry_discount = (fair_value - entry_raw) / fair_value
    entry_discount = max(0.03, min(0.15, entry_discount))
    entry_price = round(fair_value * (1 - entry_discount), 2)

    return entry_price, round(entry_discount, 4)


def _calculate_exit(
    entry_price: float,
    hist_pes: list[float],
) -> tuple[float, float, float]:
    """Historical P/E stretch exit price above entry.

    Uses the ratio of 90th percentile to median historical P/E to determine
    how much the market historically stretches this stock's valuation.

        pe_stretch   = P/E_90th / P/E_50th
        exit_premium = 0.25 + (pe_stretch - 1.0) × 0.50, clamped [0.25, 0.50]
        exit_price   = entry × (1 + exit_premium)

    Fallback: exit_premium = 0.30 when insufficient historical data.

    Returns (exit_price, exit_premium, pe_stretch).
    """
    pe_stretch = 1.0
    exit_premium = 0.30  # fallback

    if hist_pes and len(hist_pes) > 10:
        p50 = statistics.median(hist_pes)
        sorted_pes = sorted(hist_pes)
        idx_90 = int(len(sorted_pes) * 0.90)
        p90 = sorted_pes[min(idx_90, len(sorted_pes) - 1)]
        pe_stretch = p90 / p50 if p50 > 0 else 1.0

        exit_premium = 0.25 + (pe_stretch - 1.0) * 0.50
        exit_premium = max(0.25, min(0.50, exit_premium))

    exit_price = round(entry_price * (1 + exit_premium), 2)
    pe_stretch = round(pe_stretch, 2)

    return exit_price, round(exit_premium, 4), pe_stretch
