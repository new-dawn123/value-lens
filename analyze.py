import argparse
import sys

from src.data_fetcher import fetch_stock_data
from src.gates import check_gates
from src.scorer import score_stock
from src.valuator import calculate_valuation
from src.formatter import print_output, print_gate_failure


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a stock's fundamentals using PEG-based scoring"
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed scoring breakdown"
    )
    parser.add_argument(
        "--eps", type=float, default=None,
        help="Override trailing EPS with a custom value (for GAAP distortions)"
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()

    try:
        data = fetch_stock_data(ticker)
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        sys.exit(1)

    passed, gate_messages = check_gates(data)

    if not passed:
        print_gate_failure(ticker, data.get("name", "Unknown"), gate_messages)
        sys.exit(0)

    scores = score_stock(data, custom_eps=args.eps)
    valuation = calculate_valuation(data, custom_eps=args.eps)

    print_output(
        ticker=ticker,
        data=data,
        gate_messages=gate_messages,
        scores=scores,
        valuation=valuation,
        detailed=args.detailed,
    )


if __name__ == "__main__":
    main()
