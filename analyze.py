import argparse
import sys

from src.data_fetcher import fetch_stock_data
from src.gates import check_gates, check_post_valuation_gates
from src.scorer import apply_price_cap, score_stock
from src.valuator import calculate_valuation
from src.formatter import print_output, print_gate_failure


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a stock's fundamentals using PEG-based scoring"
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument(
        "--eps", type=float, default=None,
        help="Override trailing EPS with a custom value (for GAAP distortions)"
    )
    parser.add_argument(
        "--growth", type=float, default=None,
        help="Override 5Y growth rate (e.g., 15.0 for 15%%). Bypasses dampened 5Y growth."
    )
    parser.add_argument(
        "--no-hist-premium", action="store_true",
        help="Disregard Historical Premium (still shown, not applied to Fair Value)"
    )
    parser.add_argument(
        "--uncap-hist-premium", action="store_true",
        help="Remove the ±20%% clamp (0.80–1.20) on the historical premium"
    )
    args = parser.parse_args()

    ticker = args.ticker.upper().strip()

    try:
        data = fetch_stock_data(ticker)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        sys.exit(1)

    passed, gate_messages = check_gates(data, custom_growth=args.growth)

    if not passed:
        print_gate_failure(ticker, data.get("name", "Unknown"), gate_messages)
        sys.exit(0)

    scores = score_stock(data, custom_eps=args.eps, custom_growth=args.growth)
    valuation = calculate_valuation(
        data, custom_eps=args.eps, custom_growth=args.growth, scores=scores,
        disregard_hist_premium=args.no_hist_premium,
        uncap_hist_premium=args.uncap_hist_premium,
    )
    post_passed, post_messages = check_post_valuation_gates(valuation)
    if not post_passed:
        print_gate_failure(ticker, data.get("name", "Unknown"), post_messages)
        sys.exit(0)

    scores = apply_price_cap(scores, data, valuation)

    print_output(
        ticker=ticker,
        data=data,
        gate_messages=gate_messages,
        scores=scores,
        valuation=valuation,
        disregard_hist_premium=args.no_hist_premium,
    )


if __name__ == "__main__":
    main()
