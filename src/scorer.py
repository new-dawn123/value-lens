from __future__ import annotations

import math

# P/E that a zero-growth company deserves (~perpetuity at 7-8% discount rate).
BASE_PE = 12.0

# Delayed dampening: no compression up to _GROWTH_DAMPEN_THRESHOLD,
# then log-dampen the excess with k=_GROWTH_DAMPEN_K.
# t=15,k=30: 15%->15.0%, 25%->23.6%, 30%->27.2%, 50%->38.2%, 100%->55.3%.
_GROWTH_DAMPEN_THRESHOLD = 15.0
_GROWTH_DAMPEN_K = 30.0


def _fair_pe(growth: float) -> float:
    """Compute fair P/E from growth using the 5-year compounding model.

    Fair P/E = BASE_PE * (1 + g/100)^5

    Meaning: pay today's no-growth multiple for the earnings the company
    will have in 5 years. Works for negative, zero, and positive growth.
    """
    return BASE_PE * (1 + growth / 100) ** 5


def score_stock(
    data: dict,
    custom_eps: float | None = None,
    custom_growth: float | None = None,
) -> dict:
    """Score a stock on a 0-100 scale using weighted fundamental metrics.

    Returns a dict with individual metric scores, weights, and the final score.
    Custom growth is dampened the same way as fetched growth.
    """
    eps = custom_eps if custom_eps is not None else data.get("trailing_eps", 0)
    trailing_pe = data.get("trailing_pe")

    # Growth: use custom override (dampened) or compute dampened 5Y
    if custom_growth is not None:
        effective_growth = _dampen_growth(custom_growth) if custom_growth > 0 else custom_growth
    else:
        effective_growth = _compute_effective_growth(data)

    # Compute PEG: Actual P/E / Fair P/E
    if trailing_pe and effective_growth is not None:
        if custom_eps is not None and data.get("current_price"):
            actual_pe = data["current_price"] / custom_eps
        else:
            actual_pe = trailing_pe
        fp = _fair_pe(effective_growth)
        peg = actual_pe / fp if fp > 0 else None
    else:
        peg = None

    # Score each metric
    peg_score = _score_peg(peg)
    revision_score = _score_eps_revisions(data.get("eps_revisions"))
    surprise_score = _score_earnings_surprises(data.get("earnings_history"))

    # Expose effective growth used for PEG (dampened 5Y or custom)
    blended_growth = effective_growth

    # Weighted combination
    weights = {
        "peg": 0.70,
        "eps_revisions": 0.20,
        "earnings_surprises": 0.10,
    }

    weighted_sum = (
        peg_score * weights["peg"]
        + revision_score * weights["eps_revisions"]
        + surprise_score * weights["earnings_surprises"]
    )

    final_score = round(weighted_sum * 10)  # Scale 0-10 weighted → 0-100
    final_score = max(0, min(100, final_score))

    label = _score_label(final_score)

    return {
        "final_score": final_score,
        "label": label,
        "peg": peg,
        "blended_growth": blended_growth,
        "breakdown": {
            "peg": {"score": peg_score, "weight": weights["peg"]},
            "eps_revisions": {"score": revision_score, "weight": weights["eps_revisions"]},
            "earnings_surprises": {"score": surprise_score, "weight": weights["earnings_surprises"]},
        },
    }


def _dampen_growth(
    growth: float,
    threshold: float = _GROWTH_DAMPEN_THRESHOLD,
    k: float = _GROWTH_DAMPEN_K,
) -> float:
    """Delayed log-dampen: no compression up to *threshold*, then compress excess.

    For growth <= threshold:  output = growth  (trusted as-is)
    For growth > threshold:   output = threshold + k * ln(1 + (growth - threshold) / k)

    Examples with t=15, k=30:
        10% -> 10.0%,  15% -> 15.0%,  25% -> 23.6%,
        30% -> 27.2%,  50% -> 38.2%,  100% -> 55.3%

    Negative growth is returned as-is (no compression needed for pessimism).
    """
    if growth <= 0:
        return growth
    if growth <= threshold:
        return growth
    excess = growth - threshold
    return threshold + k * math.log(1 + excess / k)


def _compute_effective_growth(data: dict) -> float | None:
    """Compute effective growth for PEG: dampened 5Y estimate.

    Uses only the long-term 5Y growth estimate with delayed dampening:
    growth up to 15% is trusted as-is, excess above 15% is log-dampened
    (k=30) to compress aggressive analyst projections.

    Supports negative growth — the compounding model handles it naturally.
    Returns None if no 5Y growth data is available.
    """
    growth_5y = data.get("growth_5y")

    if growth_5y is None:
        return None

    if growth_5y > 0:
        return _dampen_growth(growth_5y)

    # Negative growth: pass through undampened
    return growth_5y


def _score_peg(peg: float | None) -> float:
    """Score PEG ratio on 0-10 scale."""
    if peg is None:
        return 5.0  # neutral if unavailable
    if peg < 0.5:
        return 10.0
    if peg < 0.75:
        return 9.0
    if peg < 1.0:
        return 7.0
    if peg < 1.25:
        return 5.0
    if peg < 1.5:
        return 4.0
    if peg < 2.0:
        return 2.0
    return 0.0


def _score_eps_revisions(revisions: dict | None) -> float:
    """Score EPS revision momentum on 0-10 scale.

    Aggregates net upward-minus-downward revisions across periods and timeframes.
    """
    if not revisions:
        return 5.0  # neutral if unavailable

    total_up = 0
    total_down = 0
    for period_data in revisions.values():
        total_up += period_data.get("up_7d", 0) + period_data.get("up_30d", 0)
        total_down += period_data.get("down_7d", 0) + period_data.get("down_30d", 0)

    total = total_up + total_down
    if total == 0:
        return 5.0

    # Net ratio: 1.0 = all up, -1.0 = all down, 0 = balanced
    net_ratio = (total_up - total_down) / total

    # Map [-1, 1] → [0, 10]
    return round(max(0.0, min(10.0, (net_ratio + 1.0) * 5.0)), 1)


def _score_earnings_surprises(earnings_history: list[dict] | None) -> float:
    """Score earnings surprise consistency on 0-10 scale.

    Based on how many of the last 4 quarters beat estimates.
    """
    if not earnings_history:
        return 5.0  # neutral if unavailable

    beats = 0
    counted = 0
    for quarter in earnings_history[-4:]:
        est = quarter.get("eps_estimate")
        actual = quarter.get("eps_actual")
        if est is not None and actual is not None:
            counted += 1
            if actual >= est:
                beats += 1

    if counted == 0:
        return 5.0

    beat_ratio = beats / counted
    # 4/4 → 10, 3/4 → 7, 2/4 → 5, 1/4 → 3, 0/4 → 0
    score_map = {4: 10.0, 3: 7.0, 2: 5.0, 1: 3.0, 0: 0.0}
    return score_map.get(beats, round(beat_ratio * 10, 1))


def apply_price_cap(scores: dict, data: dict, valuation: dict) -> dict:
    """Cap the final score based on price position relative to entry/exit.

    Gradual taper:
      - Price <= entry:  cap = 100 (no restriction)
      - Entry < price < exit:  cap tapers linearly from 100 to 50
      - Price >= exit:  cap = 50

    Returns an updated copy of the scores dict with capped final_score and label.
    The breakdown retains raw component scores for transparency.
    """
    price = data.get("current_price")
    entry = valuation.get("entry_price")
    exit_ = valuation.get("exit_price")

    if price is None or entry is None or exit_ is None:
        return scores

    if price <= entry:
        cap = 100
    elif price >= exit_:
        cap = 49
    else:
        ratio = (price - entry) / (exit_ - entry)
        cap = 100 - (ratio * 51)

    raw_score = scores["final_score"]
    capped_score = max(0, min(int(cap), raw_score))

    result = dict(scores)
    result["raw_score"] = raw_score
    result["price_cap"] = round(cap)
    result["final_score"] = capped_score
    result["label"] = _score_label(capped_score)

    return result


def _score_label(score: int) -> str:
    if score >= 80:
        return "Strong Buy"
    if score >= 70:
        return "Attractive"
    if score >= 50:
        return "Hold / Fair Value"
    if score >= 20:
        return "Unattractive"
    return "Avoid"
