"""Batch-analyze the top 500 global stocks using the ValueLens pipeline.

Fetches the largest global companies from the finviz screener (sorted by
market cap), then runs each ticker through the full scoring/valuation
pipeline.  We fetch extra tickers (~700) to guarantee exactly 500 results
with OK status after gate filtering.  Results are saved incrementally to a
CSV so the run can be resumed after interruption or rate-limit stops.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from src.data_fetcher import fetch_stock_data
from src.gates import check_gates, check_post_valuation_gates
from src.scorer import apply_price_cap, score_stock
from src.valuator import calculate_valuation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds: 2, 4, 8
_FINVIZ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}
_FINVIZ_TIMEOUT = 20

_SCREENER_URL = "https://finviz.com/screener.ashx"
_SCREENER_PARAMS = {"v": "111", "o": "-marketcap"}
_CONSECUTIVE_FAIL_LIMIT = 5
_FETCH_LIMIT = 700     # fetch extra to guarantee 500 OK results
_TARGET_OK = 500       # stop writing after this many OK rows

CSV_COLUMNS = [
    "#",
    "Ticker",
    "Name",
    "Market Cap",
    "Current Price",
    "Fair Value",
    "% vs Fair Value",
    "Fair Price",
    "% vs Fair Price",
    "Score",
]


# ---------------------------------------------------------------------------
# Global ticker list retrieval via finviz screener
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


def fetch_top_tickers(limit: int = _FETCH_LIMIT) -> list[dict]:
    """Scrape the finviz screener for the top global tickers by market cap.

    Returns a list of dicts with keys: ticker, name, sector, market_cap.
    Fetches up to *limit* tickers (default ~700) so that after gate
    filtering we can still fill 500 OK rows.
    """
    tickers: list[dict] = []
    start = 1

    while len(tickers) < limit:
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
    return tickers[:limit]


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
            # Skip leading comment lines (e.g. "# Last run: ...") but keep
            # the header row whose first column is named "#".
            while True:
                pos = f.tell()
                line = f.readline()
                if not line or not line.startswith("# "):
                    f.seek(pos)
                    break
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get("Ticker", "").strip()
                if ticker:
                    done.add(ticker)
    except Exception:
        pass
    return done


def _write_csv_header(path: str) -> None:
    """Write the CSV header row (with a timestamp comment on line 1)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Last run: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)


def _append_csv_row(path: str, row: list) -> None:
    """Append a single row to the CSV."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _count_csv_rows(path: str) -> int:
    """Count data rows in an existing CSV (for resume # numbering)."""
    if not os.path.exists(path):
        return 0
    count = 0
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            for line in f:
                # Skip comment lines ("# Last run: ...") but not the header
                # row which starts with "#," (the column name is "#").
                if line.startswith("# "):
                    continue
                if line.strip():
                    count += 1
        return max(count - 1, 0)  # subtract header row
    except Exception:
        return 0


def _format_market_cap(value) -> str:
    """Format market cap number to human-readable string."""
    if value is None:
        return ""
    try:
        v = float(value)
    except (ValueError, TypeError):
        return ""
    if v >= 1e12:
        scaled = v / 1e12
        return f"{scaled:.1f}T" if scaled >= 10 else f"{scaled:.2f}T"
    if v >= 1e9:
        scaled = v / 1e9
        # After rounding, check if it should be promoted to T
        rounded = round(scaled, 1 if scaled >= 10 else 2)
        if rounded >= 1000:
            return f"{v / 1e12:.2f}T"
        return f"{scaled:.1f}B" if scaled >= 10 else f"{scaled:.2f}B"
    if v >= 1e6:
        scaled = v / 1e6
        rounded = round(scaled, 1 if scaled >= 10 else 2)
        if rounded >= 1000:
            return f"{v / 1e9:.2f}B"
        return f"{scaled:.1f}M" if scaled >= 10 else f"{scaled:.2f}M"
    return f"{v:.0f}"


# ---------------------------------------------------------------------------
# Single-ticker analysis
# ---------------------------------------------------------------------------

def analyze_ticker(ticker: str, screener_market_cap: float | None = None) -> list | None:
    """Run the full ValueLens pipeline on a single ticker.

    Returns a list matching CSV_COLUMNS (without the '#' position, which is
    filled in by the main loop), or None if the ticker should be skipped
    (gate failure, negative prices, etc.).

    If *screener_market_cap* is provided (from the finviz screener), it is
    used for the CSV market-cap column so ordering stays consistent.
    """
    data = fetch_stock_data(ticker)

    passed, gate_messages = check_gates(data)
    if not passed:
        return None

    scores = score_stock(data)
    valuation = calculate_valuation(data, scores=scores)

    post_passed, post_messages = check_post_valuation_gates(valuation)
    if not post_passed:
        return None

    scores = apply_price_cap(scores, data, valuation)

    current_price = data.get("current_price")
    fair_value = valuation.get("peg_method", {}).get("fair_value")
    fair_price = valuation.get("fair_price")

    pct_vs_fv = ""
    if current_price and fair_value:
        pct_vs_fv = round((current_price - fair_value) / fair_value * 100, 1)

    pct_vs_fp = ""
    if current_price and fair_price:
        pct_vs_fp = round((current_price - fair_price) / fair_price * 100, 1)

    return [
        ticker,
        data.get("name", ""),
        _format_market_cap(screener_market_cap or data.get("market_cap")),
        current_price or "",
        round(fair_value, 2) if fair_value else "",
        pct_vs_fv,
        round(fair_price, 2) if fair_price else "",
        pct_vs_fp,
        scores.get("final_score", ""),
    ]


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

def _process_ticker(
    ticker: str, order_key: int, screener_market_cap: float | None = None,
) -> tuple[int, str, list | None, str]:
    """Worker function: analyze a single ticker with retries.

    Returns (order_key, ticker, row_or_None, status_str).
    row does NOT include the '#' column — that is assigned at flush time.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            row = analyze_ticker(ticker, screener_market_cap=screener_market_cap)
            if row is None:
                return order_key, ticker, None, "SKIPPED"
            return order_key, ticker, row, "OK"
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE ** (attempt + 1))
            last_error = str(e)
    return order_key, ticker, None, f"FAILED: {last_error[:100]}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch-analyze top global stocks with ValueLens"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Target number of OK results (default: 500)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds to wait between submitting tickers (default: 0.5)"
    )
    parser.add_argument(
        "--output", type=str, default="top500_valuation.csv",
        help="Output CSV file path (default: top500_valuation.csv)"
    )
    args = parser.parse_args()

    target_ok = args.limit or _TARGET_OK
    # Fetch extra tickers so we can hit the target after gate filtering
    fetch_count = int(target_ok * _FETCH_LIMIT / _TARGET_OK)

    # --- Fetch global ticker list ---
    print(f"Fetching top {fetch_count} global companies from finviz...")
    tickers_list = fetch_top_tickers(limit=fetch_count)
    if not tickers_list:
        print("Error: could not retrieve ticker list from finviz.")
        sys.exit(1)
    print(f"  Found {len(tickers_list)} tickers.")

    # --- Resume support ---
    already_done = _read_existing_csv(args.output)
    if already_done:
        remaining = [t for t in tickers_list if t["ticker"] not in already_done]
        print(f"  Resuming: {len(already_done)} already processed, "
              f"{len(remaining)} remaining.")
    else:
        remaining = tickers_list
        _write_csv_header(args.output)

    csv_row_num_start = _count_csv_rows(args.output)
    if csv_row_num_start >= target_ok:
        print(f"Already have {csv_row_num_start} OK results (target: {target_ok}). Nothing to do.")
        sys.exit(0)

    if not remaining:
        print("All tickers already processed. Nothing to do.")
        sys.exit(0)

    # Ordered list of indices for flush tracking (one per remaining ticker)
    remaining_keys = list(range(len(remaining)))

    total = len(tickers_list)
    done_count = len(already_done)
    consecutive_failures = 0
    print_lock = threading.Lock()
    stop_event = threading.Event()

    # Buffer for ordered writes: order_key → row (None = skipped/failed)
    pending: dict[int, list | None] = {}
    next_write_idx = 0  # index into remaining_keys for next flush
    csv_row_num = _count_csv_rows(args.output)  # rows already written

    def _flush_pending():
        """Write all consecutive ready results to CSV in order, assigning #.

        Returns True if we have reached the target number of OK results.
        """
        nonlocal next_write_idx, csv_row_num
        while next_write_idx < len(remaining_keys):
            key = remaining_keys[next_write_idx]
            if key not in pending:
                break
            row = pending.pop(key)
            if row is not None:
                csv_row_num += 1
                row.insert(0, csv_row_num)
                _append_csv_row(args.output, row)
                if csv_row_num >= target_ok:
                    return True
            next_write_idx += 1
        return False

    print(f"  Processing {len(remaining)} tickers with {args.workers} workers "
          f"(target: {target_ok} OK results)...\n")

    # --- Parallel process loop ---
    # Submit futures in a background thread so we can start processing
    # results immediately instead of blocking on the full submission loop.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict = {}
        futures_lock = threading.Lock()

        def _submit_all():
            for idx, entry in enumerate(remaining):
                if stop_event.is_set():
                    break
                ticker = entry["ticker"]
                future = executor.submit(
                    _process_ticker, ticker, idx, entry.get("market_cap"),
                )
                with futures_lock:
                    futures[future] = ticker
                time.sleep(args.delay)  # stagger submissions to avoid burst

        handled: set = set()
        submitter = threading.Thread(target=_submit_all, daemon=True)
        submitter.start()

        # Process results as they complete, while submission continues
        while True:
            submitter_alive = submitter.is_alive()

            # Grab a snapshot of current futures
            with futures_lock:
                snapshot = dict(futures)

            # Find completed futures not yet handled
            done_futures = [f for f in snapshot if f.done() and f not in handled]

            if not done_futures and not submitter_alive:
                # Check if all submitted futures have been handled
                with futures_lock:
                    snapshot = dict(futures)
                unhandled = [f for f in snapshot if f not in handled]
                if not unhandled:
                    break

            if not done_futures:
                time.sleep(0.1)
                continue

            for future in done_futures:
                if stop_event.is_set():
                    break

                handled.add(future)
                order_key, ticker, row, status = future.result()
                done_count += 1

                with print_lock:
                    print(f"[{done_count}/{total}] {ticker}: {status}  "
                          f"({csv_row_num}/{target_ok} OK)")

                # Buffer result and flush any consecutive ready rows
                pending[order_key] = row
                reached_target = _flush_pending()

                if reached_target:
                    with print_lock:
                        print(f"\nReached target of {target_ok} OK results. Stopping.")
                    stop_event.set()
                    break

                if status.startswith("FAILED"):
                    consecutive_failures += 1
                    if consecutive_failures >= _CONSECUTIVE_FAIL_LIMIT:
                        with print_lock:
                            print(f"\n{_CONSECUTIVE_FAIL_LIMIT} consecutive failures — "
                                  f"likely rate-limited. Stopping.")
                            print(f"Progress saved to {args.output}. "
                                  f"Re-run to resume from where you left off.")
                        stop_event.set()
                        submitter.join(timeout=2)
                        sys.exit(2)
                else:
                    consecutive_failures = 0

            if stop_event.is_set():
                break

        submitter.join(timeout=2)

    # Flush any remaining buffered results (won't exceed target)
    _flush_pending()

    print(f"\nDone! {csv_row_num} results saved to {args.output}")


if __name__ == "__main__":
    main()
