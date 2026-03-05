from __future__ import annotations


def check_gates(data: dict, custom_growth: float | None = None) -> tuple[bool, list[str]]:
    """Run red flag gate checks on the fetched data.

    Returns:
        (passed, messages) where passed=False means scoring should be skipped.
        Messages include both hard failures and soft warnings.
    When custom_growth is provided, it overrides the fetched growth_5y for gate checks.
    """
    messages = []
    passed = True

    trailing_eps = data.get("trailing_eps")
    growth_5y = custom_growth if custom_growth is not None else data.get("growth_5y")
    current_price = data.get("current_price")
    trailing_pe = data.get("trailing_pe")

    # Hard gates — block scoring
    if trailing_eps is None or trailing_eps <= 0:
        messages.append("Cannot score: negative or zero trailing earnings — PEG ratio is meaningless")
        passed = False

    if growth_5y is None:
        messages.append("Cannot score: no 5-year EPS growth estimate available")
        passed = False

    if growth_5y is not None and growth_5y < 3.0:
        messages.append(
            f"PEG analysis not applicable: estimated 5Y growth is {growth_5y:.1f}% (<3%)"
        )
        passed = False

    # Soft gates — warnings only
    if data.get("fx_converted"):
        fc = data.get("financial_currency", "?")
        pc = data.get("currency", "?")
        fx = data.get("fx_rate", 0)
        messages.append(
            f"Warning: financials reported in {fc}, converted to {pc} (rate: {fx:.4f})"
        )
    elif data.get("currency_mismatch"):
        messages.append(
            "Warning: currency mismatch detected (ADR/foreign listing) — "
            "historical P/E charts may be unavailable"
        )

    if passed and trailing_pe is not None and growth_5y is not None and growth_5y > 0:
        peg = trailing_pe / growth_5y
        if peg > 3.0:
            messages.append(f"Warning: extremely high PEG ratio ({peg:.2f})")

    if current_price is not None and current_price < 5.0:
        messages.append("Warning: penny stock territory (price < $5)")

    return passed, messages
