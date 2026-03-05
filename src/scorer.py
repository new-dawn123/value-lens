from __future__ import annotations

import math


def score_stock(
    data: dict,
    custom_eps: float | None = None,
    custom_growth: float | None = None,
) -> dict:
    """Score a stock on a 0-100 scale using weighted fundamental metrics.

    Returns a dict with individual metric scores, weights, and the final score.
    When custom_growth is provided, it replaces the blended growth entirely
    (used as-is, no dampening).
    """
    eps = custom_eps if custom_eps is not None else data.get("trailing_eps", 0)
    growth_5y = data.get("growth_5y", 0)
    trailing_pe = data.get("trailing_pe")

    # Growth: use custom override or compute blended
    if custom_growth is not None:
        blended_growth = custom_growth
    else:
        blended_growth = _compute_blended_growth(data)

    # Compute PEG using growth
    if trailing_pe and blended_growth and blended_growth > 0:
        if custom_eps is not None and data.get("current_price"):
            peg = (data["current_price"] / custom_eps) / blended_growth
        else:
            peg = trailing_pe / blended_growth
    else:
        peg = None

    # Compute PSG: P/S / Revenue Growth (or fallback to raw P/S)
    ps_ratio = data.get("ps_ratio")
    revenue_growth = data.get("revenue_growth_next_year")
    psg = None
    using_psg = False
    if ps_ratio is not None and revenue_growth is not None and revenue_growth > 0:
        psg = ps_ratio / revenue_growth
        using_psg = True
    elif ps_ratio is not None:
        psg = ps_ratio  # fallback: raw P/S
        using_psg = False

    # Score each metric
    peg_score = _score_peg(peg)
    psg_score = _score_psg(psg, using_psg)
    revision_score = _score_eps_revisions(data.get("eps_revisions"))
    surprise_score = _score_earnings_surprises(data.get("earnings_history"))

    # Weighted combination
    weights = {
        "peg": 0.45,
        "psg": 0.30,
        "eps_revisions": 0.15,
        "earnings_surprises": 0.10,
    }

    weighted_sum = (
        peg_score * weights["peg"]
        + psg_score * weights["psg"]
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
        "psg": psg,
        "using_psg": using_psg,
        "blended_growth": blended_growth,
        "breakdown": {
            "peg": {"score": peg_score, "weight": weights["peg"]},
            "psg": {"score": psg_score, "weight": weights["psg"]},
            "eps_revisions": {"score": revision_score, "weight": weights["eps_revisions"]},
            "earnings_surprises": {"score": surprise_score, "weight": weights["earnings_surprises"]},
        },
    }


def _dampen_growth(growth: float, k: float = 15.0) -> float:
    """Log-dampen a growth rate to compress high values.

    dampen(growth, k) = k * ln(1 + growth / k)

    Examples with k=15:
        10% -> ~9.1%,  15% -> ~12.8%,  25% -> ~18.3%,  50% -> ~27.6%
    """
    if growth <= 0:
        return growth
    return k * math.log(1 + growth / k)


def _compute_blended_growth(data: dict) -> float | None:
    """Compute 3-horizon blended growth for PEG: 0y (35%) + +1y (35%) + 5y dampened (30%).

    Near-term (0y, +1y) are undampened — grounded in observable/near-term data.
    Long-term (5y) is log-dampened to compress aggressive projections.
    If any component is missing, its weight is redistributed proportionally.
    Returns None if no growth data is available at all.
    """
    growth_0y = data.get("growth_current_year")
    growth_1y = data.get("growth_next_year")
    growth_5y = data.get("growth_5y")

    # Build components: near-term raw, long-term dampened
    components = []
    if growth_0y and growth_0y > 0:
        components.append((growth_0y, 0.35))
    if growth_1y and growth_1y > 0:
        components.append((growth_1y, 0.35))
    if growth_5y and growth_5y > 0:
        components.append((_dampen_growth(growth_5y), 0.30))

    if not components:
        return None

    # Normalize weights if some components are missing
    total_weight = sum(w for _, w in components)
    return sum(val * (w / total_weight) for val, w in components)


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


def _score_psg(psg: float | None, using_psg: bool) -> float:
    """Score PSG ratio (or raw P/S as fallback) on 0-10 scale."""
    if psg is None:
        return 5.0  # neutral if unavailable

    if using_psg:
        # PSG thresholds (P/S divided by revenue growth)
        if psg < 0.5:
            return 10.0
        if psg < 1.0:
            return 8.0
        if psg < 2.0:
            return 5.0
        return 2.0
    else:
        # Raw P/S fallback thresholds
        if psg < 1.0:
            return 10.0
        if psg < 2.0:
            return 7.0
        if psg < 5.0:
            return 4.0
        if psg < 10.0:
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
