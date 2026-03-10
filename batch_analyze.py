"""Batch-analyze all S&P 500 stocks using the ValueLens pipeline.

Fetches the S&P 500 constituent list from the finviz screener, then runs
each ticker through the full scoring/valuation pipeline.  Results are saved
incrementally to a CSV so the run can be resumed after interruption or
rate-limit stops.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

from src.data_fetcher import (
    _BACKOFF_BASE,
    _FINVIZ_HEADERS,
    _FINVIZ_TIMEOUT,
    _MAX_RETRIES,
    fetch_stock_data,
)
from src.gates import check_gates
from src.scorer import apply_price_cap, score_stock
from src.valuator import calculate_valuation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SCREENER_URL = "https://finviz.com/screener.ashx"
_SCREENER_PARAMS = {"v": "111", "f": "idx_sp500", "o": "-marketcap"}
_CONSECUTIVE_FAIL_LIMIT = 5

CSV_COLUMNS = [
    "Ticker",
    "Name",
    "Sector",
    "Market Cap",
    "Current Price",
    "Fair Value",
    "Entry Price",
    "Exit Price",
    "Score",
    "Label",
    "Status",
]


# ---------------------------------------------------------------------------
# S&P 500 list retrieval via finviz screener
# ---------------------------------------------------------------------------

def _parse_market_cap(text: str) -> float:
    """Convert finviz market cap string (e.g. '3.45T', '120.5B') to a float."""
    text = text.strip()
    if not text or text == "-":
        return 0.0
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    suffix = text[-1].upper()
    if suffix in multipliers:
        try:
            return float(text[:-1]) * multipliers[suffix]
        except ValueError:
            return 0.0
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return 0.0


def _fetch_screener_page(start: int = 1) -> str | None:
    """Fetch a single page of finviz screener results with retry."""
    params = {**_SCREENER_PARAMS, "r": str(start)}
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(
                _SCREENER_URL,
                params=params,
                headers=_FINVIZ_HEADERS,
                timeout=_FINVIZ_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.text
        except Exception:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE ** (attempt + 1))
    return None


def fetch_sp500_tickers() -> list[dict]:
    """Scrape the finviz screener for all S&P 500 tickers sorted by market cap.

    Returns a list of dicts with keys: ticker, name, sector, market_cap.
    """
    tickers: list[dict] = []
    start = 1

    while True:
        html = _fetch_screener_page(start)
        if html is None:
            break

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", class_="screener_table")
        if table is None:
            # Try alternative class name
            table = soup.find("table", {"id": "screener-content"})
        if table is None:
            # Fallback: find table by looking at rows with ticker-like content
            tables = soup.find_all("table")
            for t in tables:
                rows = t.find_all("tr")
                if len(rows) > 2:
                    cells = rows[1].find_all("td")
                    if len(cells) >= 10:
                        table = t
                        break

        if table is None:
            break

        rows = table.find_all("tr")[1:]  # skip header
        if not rows:
            break

        page_count = 0
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 10:
                continue
            ticker = cells[1].text.strip()
            name = cells[2].text.strip()
            sector = cells[3].text.strip()
            market_cap_str = cells[6].text.strip()
            if not ticker:
                continue
            tickers.append({
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "market_cap": _parse_market_cap(market_cap_str),
            })
            page_count += 1

        if page_count < 20:
            break  # last page

        start += 20
        time.sleep(1)  # polite delay between screener pages

    # Sort by market cap descending (should already be sorted, but ensure)
    tickers.sort(key=lambda x: x["market_cap"], reverse=True)
    return tickers


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_existing_csv(path: str) -> set[str]:
    """Read already-processed tickers from an existing CSV."""
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get("Ticker", "").strip()
                if ticker:
                    done.add(ticker)
    except Exception:
        pass
    return done


def _write_csv_header(path: str) -> None:
    """Write the CSV header row."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)


def _append_csv_row(path: str, row: list) -> None:
    """Append a single row to the CSV."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _format_market_cap(value) -> str:
    """Format market cap number to human-readable string."""
    if value is None:
        return ""
    try:
        v = float(value)
    except (ValueError, TypeError):
        return ""
    if v >= 1e12:
        return f"{v / 1e12:.2f}T"
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    return f"{v:.0f}"


# ---------------------------------------------------------------------------
# Single-ticker analysis
# ---------------------------------------------------------------------------

def analyze_ticker(ticker: str) -> list:
    """Run the full ValueLens pipeline on a single ticker.

    Returns a list matching CSV_COLUMNS.
    """
    data = fetch_stock_data(ticker)

    passed, gate_messages = check_gates(data)

    if not passed:
        reason = "; ".join(gate_messages) if gate_messages else "Unknown gate"
        return [
            ticker,
            data.get("name", ""),
            "",
            _format_market_cap(data.get("market_cap")),
            data.get("current_price", ""),
            "",  # Fair Value
            "",  # Entry Price
            "",  # Exit Price
            "",  # Score
            "",  # Label
            f"Gate Failed: {reason}",
        ]

    scores = score_stock(data)
    valuation = calculate_valuation(data, scores=scores)
    scores = apply_price_cap(scores, data, valuation)

    return [
        ticker,
        data.get("name", ""),
        "",
        _format_market_cap(data.get("market_cap")),
        data.get("current_price", ""),
        valuation.get("fair_value", ""),
        valuation.get("entry_price", ""),
        valuation.get("exit_price", ""),
        scores.get("final_score", ""),
        scores.get("label", ""),
        "OK",
    ]


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch-analyze S&P 500 stocks with ValueLens"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the top N tickers by market cap (default: all)"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds to wait between tickers (default: 2)"
    )
    parser.add_argument(
        "--output", type=str, default="sp500_valuation.csv",
        help="Output CSV file path (default: sp500_valuation.csv)"
    )
    args = parser.parse_args()

    # --- Fetch S&P 500 list ---
    print("Fetching S&P 500 constituent list from finviz...")
    sp500 = fetch_sp500_tickers()
    if not sp500:
        print("Error: could not retrieve S&P 500 list from finviz.")
        sys.exit(1)
    print(f"  Found {len(sp500)} tickers.")

    # Apply limit
    if args.limit:
        sp500 = sp500[: args.limit]
        print(f"  Limited to top {args.limit} by market cap.")

    # --- Resume support ---
    already_done = _read_existing_csv(args.output)
    if already_done:
        remaining = [t for t in sp500 if t["ticker"] not in already_done]
        print(f"  Resuming: {len(already_done)} already processed, "
              f"{len(remaining)} remaining.")
    else:
        remaining = sp500
        _write_csv_header(args.output)

    if not remaining:
        print("All tickers already processed. Nothing to do.")
        sys.exit(0)

    # --- Process loop ---
    total = len(sp500)
    done_count = len(already_done)
    consecutive_failures = 0

    for entry in remaining:
        ticker = entry["ticker"]
        done_count += 1
        print(f"[{done_count}/{total}] Analyzing {ticker}...", end=" ", flush=True)

        success = False
        last_error = ""

        for attempt in range(_MAX_RETRIES):
            try:
                row = analyze_ticker(ticker)
                # Fill in sector from screener data
                row[2] = entry["sector"]
                _append_csv_row(args.output, row)
                status = row[-1]
                print(f"{status}")
                success = True
                consecutive_failures = 0
                break
            except Exception as e:
                last_error = str(e)
                if attempt < _MAX_RETRIES - 1:
                    wait = _BACKOFF_BASE ** (attempt + 1)
                    print(f"retry({attempt + 1})...", end=" ", flush=True)
                    time.sleep(wait)

        if not success:
            consecutive_failures += 1
            error_msg = last_error[:100] if last_error else "Unknown error"
            print(f"FAILED: {error_msg}")
            # Write error row
            _append_csv_row(args.output, [
                ticker, entry["name"], entry["sector"],
                "", "", "", "", "", "", "",
                f"Error: {error_msg}",
            ])

            if consecutive_failures >= _CONSECUTIVE_FAIL_LIMIT:
                print(f"\n{_CONSECUTIVE_FAIL_LIMIT} consecutive failures — "
                      f"likely rate-limited. Stopping.")
                print(f"Progress saved to {args.output}. "
                      f"Re-run to resume from where you left off.")
                sys.exit(2)

        if done_count < total:
            time.sleep(args.delay)

    print(f"\nDone! Results saved to {args.output}")


if __name__ == "__main__":
    main()
