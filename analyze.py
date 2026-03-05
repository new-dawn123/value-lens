import argparse
import sys

from src.data_fetcher import fetch_stock_data
from src.gates import check_gates
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
    args = parser.parse_args()

    ticker = args.ticker.upper()

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
        data, custom_eps=args.eps, custom_growth=args.growth, scores=scores
    )
    scores = apply_price_cap(scores, data, valuation)

    print_output(
        ticker=ticker,
        data=data,
        gate_messages=gate_messages,
        scores=scores,
        valuation=valuation,
    )


if __name__ == "__main__":
    main()
