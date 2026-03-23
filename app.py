import base64
import html as html_mod
import statistics
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_fetcher import fetch_stock_data
from src.gates import check_gates, check_post_valuation_gates
from src.scorer import apply_price_cap, score_stock
from src.valuator import (
    calculate_valuation,
    compute_historical_forward_pe_series,
    compute_historical_pe_series,
)

_PROJECT_DIR = Path(__file__).parent
_LOGO_PATH = _PROJECT_DIR / "assets" / "logo.png"
_CSV_PATH = _PROJECT_DIR / "top500_valuation.csv"


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(ticker: str, cache_version: int = 0) -> dict:
    """Cache stock data for 1 hour.

    cache_version is incremented when finviz failed on a previous fetch,
    so the next submit bypasses the stale cache and retries finviz.
    """
    return fetch_stock_data(ticker)


# ---------------------------------------------------------------------------
# Batch display helpers
# ---------------------------------------------------------------------------

def _pct_to_bg(val: float) -> str:
    """Map a percentage to a green-yellow-red background.

    Green at -10% or below, yellow at +15%, red at +40% or above.
    """
    if pd.isna(val):
        return ""
    MIN_V, MAX_V = -10.0, 40.0
    MID_V = (MIN_V + MAX_V) / 2.0  # 15.0
    clamped = max(MIN_V, min(MAX_V, float(val)))

    # Interpolate color: green(76,175,80) → yellow(255,235,59) → red(244,67,54)
    if clamped <= MID_V:
        t = (clamped - MIN_V) / (MID_V - MIN_V)  # 0..1
        r = int(76 + t * (255 - 76))
        g = int(175 + t * (235 - 175))
        b = int(80 + t * (59 - 80))
    else:
        t = (clamped - MID_V) / (MAX_V - MID_V)  # 0..1
        r = int(255 + t * (244 - 255))
        g = int(235 + t * (67 - 235))
        b = int(59 + t * (54 - 59))

    return f"color: rgb({r},{g},{b})"


def _apply_valuation_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """Return a same-shaped DataFrame of CSS styles with green/yellow/red gradients."""
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for pct_col, price_col in [
        ("% vs Fair Value", "Fair Value"),
        ("% vs Fair Price", "Fair Price"),
    ]:
        if pct_col in df.columns:
            styles[pct_col] = df[pct_col].map(_pct_to_bg)
        if price_col in df.columns and pct_col in df.columns:
            styles[price_col] = df[pct_col].map(_pct_to_bg)
    return styles


def _read_batch_run_date() -> str | None:
    """Read the 'Last run' timestamp from the CSV comment header."""
    if not _CSV_PATH.exists():
        return None
    try:
        with open(_CSV_PATH, "r", encoding="utf-8") as f:
            first = f.readline()
        if first.startswith("# Last run:"):
            return first.removeprefix("# Last run:").strip()
    except Exception:
        pass
    return None


def _load_batch_csv() -> pd.DataFrame | None:
    """Load and prepare the batch CSV. Returns None if unavailable."""
    if not _CSV_PATH.exists():
        return None

    try:
        # Skip leading comment lines (e.g. "# Last run: ...") but keep the
        # header row which also starts with "#" (the column name).
        skip = 0
        with open(_CSV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("# "):
                    skip += 1
                else:
                    break
        df = pd.read_csv(_CSV_PATH, skiprows=skip, on_bad_lines="skip")
    except Exception:
        return None

    if df.empty:
        return None

    # Coerce numeric columns (empty strings from gate failures become NaN)
    numeric_cols = [
        "Current Price", "Fair Value", "% vs Fair Value",
        "Fair Price", "% vs Fair Price", "Score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# --- Page config ---
st.set_page_config(page_title="ValueLens", page_icon=str(_LOGO_PATH), layout="wide")

st.html(
    "<style>"
    "section.main > div.block-container,"
    "[data-testid='stMainBlockContainer'],"
    "[data-testid='stAppViewBlockContainer'] {"
    "  padding-left: 15% !important;"
    "  padding-right: 15% !important;"
    "  max-width: 100% !important;"
    "}"
    "</style>"
)

_logo_b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
st.markdown(
    f"<link href='https://fonts.googleapis.com/css2?family=Inter:wght@600&display=swap'"
    f" rel='stylesheet'>"
    f"<div style='display:flex;align-items:center;gap:14px;margin-bottom:4px'>"
    f"<img src='data:image/png;base64,{_logo_b64}'"
    f" style='width:56px;height:56px;object-fit:contain'>"
    f"<div>"
    f"<span style='font-family:Inter,sans-serif;font-size:2.1em;font-weight:600;"
    f"line-height:1;letter-spacing:-0.02em'>"
    f"<span style='color:#1c6295'>Value</span>"
    f"<span style='color:#50a375'>Lens</span></span><br>"
    f"<span style='color:gray;font-size:0.9em'>Stock Fundamentals Analyzer</span>"
    f"</div></div>",
    unsafe_allow_html=True,
)

# --- Tabs ---
tab_analyzer, tab_batch = st.tabs(["Analyzer", "Batch Analysis"])

# ===== Analyzer tab =====
with tab_analyzer:

    # --- Handle deep-link from batch table ---
    _qp_ticker = st.query_params.get("ticker", "")

    # Seed the text input from the query-param only once so that
    # subsequent reruns never overwrite what the user typed.
    if _qp_ticker and "_ticker_seeded" not in st.session_state:
        st.session_state["_ticker_input"] = _qp_ticker
        st.session_state["_ticker_seeded"] = True

    # --- Section 1: Input bar ---
    with st.form("analyze_form"):
        col_ticker, col_button = st.columns([3, 1])
        with col_ticker:
            ticker = st.text_input(
                "Ticker", key="_ticker_input", placeholder="e.g. NVDA",
                label_visibility="collapsed",
            )
        with col_button:
            analyze = st.form_submit_button("Analyze", type="primary", width="stretch")

    with st.expander("Advanced Options"):
        # Pull actual values from last result for defaults
        _prev = st.session_state.get("result", {}).get("data", {})
        _default_eps = round(_prev["trailing_eps"], 2) if _prev.get("trailing_eps") else 1.00
        _default_growth = round(_prev.get("growth_5y") or 15.0, 1)

        use_custom_eps = st.checkbox(
            "Override trailing EPS",
            help="Override trailing EPS for stocks with GAAP distortions.",
        )
        custom_eps = None
        if use_custom_eps:
            custom_eps = st.number_input(
                "Custom EPS",
                min_value=0.01,
                value=_default_eps,
                step=0.01,
                format="%.2f",
                placeholder="e.g. 5.67",
            )
        use_custom_growth = st.checkbox(
            "Override 5Y growth rate",
            help="Bypasses dampened 5Y growth calculation.",
        )
        custom_growth = None
        if use_custom_growth:
            custom_growth = st.number_input(
                "Custom 5Y Growth (%)",
                min_value=0.1,
                value=_default_growth,
                step=0.5,
                format="%.1f",
                placeholder="e.g. 15.0",
            )
        disregard_hist_premium = st.checkbox(
            "Disregard Historical Premium",
            help="Still calculated and shown, but not applied to Fair Price.",
        )
        uncap_hist_premium = st.checkbox(
            "Uncap Historical Premium",
            help="Remove the \u00b120% clamp (0.80\u20131.20) on the historical premium.",
        )

    # Auto-analyze: deep-link first visit, or re-run when options change
    _auto_done = st.session_state.get("_auto_analyzed_ticker")
    _opts_fingerprint = (use_custom_eps, custom_eps, use_custom_growth,
                         custom_growth, disregard_hist_premium, uncap_hist_premium)
    _prev_fingerprint = st.session_state.get("_opts_fingerprint")
    if _qp_ticker and not analyze:
        _is_deep_link = _auto_done != _qp_ticker
        _opts_changed = ("result" in st.session_state
                         and _opts_fingerprint != _prev_fingerprint)
        if _is_deep_link or _opts_changed:
            analyze = True
            ticker = _qp_ticker
        if _is_deep_link:
            st.session_state["_auto_analyzed_ticker"] = _qp_ticker

    # --- Orchestration ---
    # NOTE: Do not use st.stop() — it kills the entire script and
    # prevents the Batch Analysis tab from rendering.
    if analyze and ticker:
        ticker = ticker.upper().strip()
        st.query_params["ticker"] = ticker
        _abort = False

        fetch_error = None
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                # Bust cache for tickers where finviz previously failed
                finviz_fails = st.session_state.setdefault("_finviz_fails", {})
                cache_ver = finviz_fails.get(ticker, 0)
                data = cached_fetch(ticker, cache_version=cache_ver)

                if not data.get("_finviz_ok", True):
                    # Next submit will use a new cache key -> retry finviz
                    finviz_fails[ticker] = cache_ver + 1
                else:
                    finviz_fails.pop(ticker, None)
            except Exception as e:
                fetch_error = e

        if fetch_error is not None:
            st.error(f"Error fetching data for {ticker}: {fetch_error}")
            _abort = True

        if not _abort:
            passed, gate_messages = check_gates(data, custom_growth=custom_growth)
            if not passed:
                st.error(f"**{ticker} \u2014 {data.get('name', 'Unknown')}**")
                for msg in gate_messages:
                    st.error(msg)
                _abort = True

        if not _abort:
            scores = score_stock(data, custom_eps=custom_eps, custom_growth=custom_growth)
            valuation = calculate_valuation(
                data, custom_eps=custom_eps, custom_growth=custom_growth, scores=scores,
                disregard_hist_premium=disregard_hist_premium,
                uncap_hist_premium=uncap_hist_premium,
            )
            post_passed, post_messages = check_post_valuation_gates(valuation)
            if not post_passed:
                st.error(f"**{ticker} — {data.get('name', 'Unknown')}**")
                for msg in post_messages:
                    st.error(msg)
                _abort = True

        if _abort:
            st.session_state.pop("result", None)
        else:
            scores = apply_price_cap(scores, data, valuation)
            pe_series = compute_historical_pe_series(data)
            fwd_pe_series = compute_historical_forward_pe_series(data)

            st.session_state["result"] = {
                "ticker": ticker,
                "data": data,
                "scores": scores,
                "valuation": valuation,
                "pe_series": pe_series,
                "fwd_pe_series": fwd_pe_series,
                "gate_messages": gate_messages,
                "disregard_hist_premium": disregard_hist_premium,
            }
            st.session_state["_opts_fingerprint"] = _opts_fingerprint

    # --- Rendering (from session state) ---
    if "result" in st.session_state:
        try:
            r = st.session_state["result"]
            ticker = r["ticker"]
            data = r["data"]
            scores = r["scores"]
            valuation = r["valuation"]
            pe_series = r["pe_series"]
            fwd_pe_series = r["fwd_pe_series"]
            disregard_hist_premium = r.get("disregard_hist_premium", False)

            # --- Section 2: Score & Summary ---
            name = data.get("name", ticker)
            price = data.get("current_price")

            website = data.get("website")
            logo_url = ""
            if website:
                domain = urlparse(website).netloc or urlparse(website).path
                logo_url = f"https://www.google.com/s2/favicons?domain={html_mod.escape(domain, quote=True)}&sz=128"

            if logo_url:
                _esc_ticker = html_mod.escape(ticker)
                _esc_name = html_mod.escape(name)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:12px'>"
                    f"<img src='{logo_url}' width='40' height='40' style='border-radius:6px'"
                    f" onerror=\"this.style.display='none'\">"
                    f"<span style='font-size:1.5em;font-weight:600'>{_esc_ticker} \u2014 {_esc_name}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"### {ticker} \u2014 {name}")
            score = scores["final_score"]
            label = scores["label"]

            score_colors = {
                "Strong Buy": "green", "Attractive": "green",
                "Hold": "orange", "Unattractive": "red", "Avoid": "red",
            }
            score_color = score_colors.get(label, "gray")
            _esc_label = html_mod.escape(label)
            st.markdown(
                f"<h2 style='color:{score_color};margin-top:0'>{score}/100 \u2014 {_esc_label}</h2>",
                unsafe_allow_html=True,
            )

            # Metric cards — HTML flexbox for full-width space-between
            peg_m_top = valuation.get("peg_method", {})
            hist_p_top = valuation.get("historical_premium", {})
            net_cash_top = valuation.get("net_cash_per_share")

            def _metric_card(label_txt, value_txt, deltas=None):
                """Return HTML for a single metric card matching st.metric style."""
                _l = html_mod.escape(str(label_txt))
                _v = html_mod.escape(str(value_txt))
                card = (
                    f"<div>"
                    f"<div style='font-size:0.9rem;color:var(--text-color, rgba(224,228,238,0.6));margin-bottom:4px'>{_l}</div>"
                    f"<div style='font-size:2.25rem;font-weight:600;line-height:1.2'>{_v}</div>"
                )
                for d in (deltas or []):
                    arrow = "" if d.get("no_arrow") else ("&#x2193; " if d.get("down") else "&#x2191; ")
                    _dt = html_mod.escape(str(d['text']))
                    card += (
                        f"<div style='font-size:0.85rem;color:var(--text-color, rgba(224,228,238,0.7));"
                        f"background:rgba(255,255,255,0.08);border-radius:16px;"
                        f"padding:2px 8px;margin-top:4px;display:inline-block'>"
                        f"{arrow}{_dt}</div><br>"
                    )
                card += "</div>"
                return card

            cp_val = f"${price:,.2f}" if price else "N/A"
            fv_raw = peg_m_top.get("fair_value")
            fv_val = f"${fv_raw:,.2f}" if fv_raw else "N/A"

            fp = valuation.get("fair_price")
            fp_val = f"${fp:,.2f}" if fp else "N/A"
            fp_deltas = []
            premium_top = hist_p_top.get("premium")
            raw_premium_top = hist_p_top.get("raw_premium")
            if premium_top is not None:
                is_capped = raw_premium_top is not None and round(raw_premium_top, 2) != round(premium_top, 2)
                prem_pct = abs(round((premium_top - 1) * 100))
                txt = f"{prem_pct}%"
                if is_capped:
                    raw_pct = abs(round((raw_premium_top - 1) * 100))
                    txt += f" ({raw_pct}%)"
                txt += " premium"
                if not disregard_hist_premium:
                    fp_deltas.append({"text": txt, "down": premium_top < 1})
            if net_cash_top is not None:
                fp_deltas.append({"text": f"${abs(net_cash_top):,.2f} net cash", "down": net_cash_top < 0})

            ep = valuation.get("entry_price")
            margin_pct = round(valuation.get("margin_of_safety", 0) * 100)
            ep_val = f"≤ ${ep:,.2f}" if ep else "N/A"

            xp = valuation.get("exit_price")
            exit_pct = round(valuation.get("exit_premium", 0) * 100)
            xp_val = f"≥ ${xp:,.2f}" if xp else "N/A"

            peg_val = scores.get("peg")
            cp_deltas = [{"text": f"PEG {peg_val:.2f}", "no_arrow": True}] if peg_val is not None else []
            eff_g = scores.get("blended_growth")
            fv_deltas = [{"text": f"Growth {eff_g:.1f}%", "no_arrow": True}] if eff_g is not None else []

            cards = (
                _metric_card("Current Price", cp_val, cp_deltas)
                + _metric_card("Fair Value", fv_val, fv_deltas)
                + _metric_card("Fair Price", fp_val, fp_deltas)
                + _metric_card("Entry Price", ep_val, [{"text": f"{margin_pct}% margin", "down": True}])
                + _metric_card("Exit Price", xp_val, [{"text": f"{exit_pct}% above entry", "down": False}])
            )
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;gap:1rem'>{cards}</div>",
                unsafe_allow_html=True,
            )

            st.divider()

            # --- Section 3: Price chart ---
            historical_prices = data.get("historical_prices")
            if historical_prices:
                dates = [p["date"] for p in historical_prices]
                prices = [p["price"] for p in historical_prices]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, y=prices, mode="lines",
                    name="Price", line=dict(color="#42A5F5", width=2),
                ))

                fv = valuation.get("fair_price")
                ep = valuation.get("entry_price")
                xp = valuation.get("exit_price")

                if ep:
                    fig.add_hline(
                        y=ep, line_dash="dash", line_color="#4DB6AC",
                        annotation_text=f"Entry ${ep:,.0f}",
                        annotation_position="bottom right",
                    )
                if fv:
                    fig.add_hline(
                        y=fv, line_dash="solid", line_color="#FFB74D",
                        annotation_text=f"Fair Price ${fv:,.0f}",
                        annotation_position="top right",
                    )
                if xp:
                    fig.add_hline(
                        y=xp, line_dash="dash", line_color="#EF5350",
                        annotation_text=f"Exit ${xp:,.0f}",
                        annotation_position="top right",
                    )

                fig.update_layout(
                    title=f"{ticker} \u2014 Price vs Fair Price",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400,
                    showlegend=False,
                )

                st.plotly_chart(fig, width="stretch")

            # --- Section 4: Historical P/E charts (side by side) ---
            pe_col_left, pe_col_right = st.columns(2)

            with pe_col_left:
                if pe_series and len(pe_series) >= 6:
                    pe_dates = [p["date"] for p in pe_series]
                    pe_values = [p["pe"] for p in pe_series]
                    median_pe = statistics.median(pe_values)

                    fig_pe = go.Figure()
                    fig_pe.add_trace(go.Scatter(
                        x=pe_dates, y=pe_values, mode="lines",
                        name="P/E", line=dict(color="#9C27B0", width=2),
                    ))
                    current_pe = data.get("trailing_pe")
                    median_pos = "top right" if not current_pe or median_pe >= current_pe else "bottom right"
                    fig_pe.add_hline(
                        y=median_pe, line_dash="dash", line_color="gray",
                        annotation_text=f"Median P/E {median_pe:.1f}",
                        annotation_position=median_pos,
                    )

                    if current_pe:
                        current_pos = "top right" if current_pe >= median_pe else "bottom right"
                        fig_pe.add_hline(
                            y=current_pe, line_dash="dot", line_color="orange",
                            annotation_text=f"Current P/E {current_pe:.1f}",
                            annotation_position=current_pos,
                        )

                    fig_pe.update_layout(
                        title=f"{ticker} \u2014 Historical P/E",
                        xaxis_title="Date",
                        yaxis_title="P/E Ratio",
                        height=350,
                        showlegend=False,
                    )

                    st.plotly_chart(fig_pe, width="stretch")
                else:
                    st.info("Historical P/E chart unavailable \u2014 insufficient positive earnings history.")

            with pe_col_right:
                if fwd_pe_series and len(fwd_pe_series) >= 6:
                    fwd_dates = [p["date"] for p in fwd_pe_series]
                    fwd_values = [p["pe"] for p in fwd_pe_series]
                    fwd_median = statistics.median(fwd_values)

                    fig_fwd = go.Figure()
                    fig_fwd.add_trace(go.Scatter(
                        x=fwd_dates, y=fwd_values, mode="lines",
                        name="Forward P/E", line=dict(color="#00897B", width=2),
                    ))
                    current_fwd_pe = data.get("forward_pe")
                    fwd_median_pos = "top right" if not current_fwd_pe or fwd_median >= current_fwd_pe else "bottom right"
                    fig_fwd.add_hline(
                        y=fwd_median, line_dash="dash", line_color="gray",
                        annotation_text=f"Median Fwd P/E {fwd_median:.1f}",
                        annotation_position=fwd_median_pos,
                    )

                    if current_fwd_pe:
                        fwd_current_pos = "top right" if current_fwd_pe >= fwd_median else "bottom right"
                        fig_fwd.add_hline(
                            y=current_fwd_pe, line_dash="dot", line_color="orange",
                            annotation_text=f"Current Fwd P/E {current_fwd_pe:.1f}",
                            annotation_position=fwd_current_pos,
                        )

                    fig_fwd.update_layout(
                        title=f"{ticker} \u2014 Historical Forward P/E",
                        xaxis_title="Date",
                        yaxis_title="Forward P/E",
                        height=350,
                        showlegend=False,
                    )

                    st.plotly_chart(fig_fwd, width="stretch")
                else:
                    st.info("Historical Forward P/E chart unavailable \u2014 insufficient earnings data.")

            # --- Section 5: Detailed Metrics (expandable) ---
            with st.expander("Detailed Metrics"):

                peg_m = valuation.get("peg_method", {})
                hist_p = valuation.get("historical_premium", {})

                def _html_table(title: str, metrics: dict) -> str:
                    """Build an HTML metric table block."""
                    _esc = html_mod.escape
                    rows = "".join(
                        f"<tr><td style='padding:4px 8px'>{_esc(str(k))}</td>"
                        f"<td style='padding:4px 8px;text-align:right'>{_esc(str(v))}</td></tr>"
                        for k, v in metrics.items()
                    )
                    return (
                        f"<div style='break-inside:avoid;margin-bottom:24px'>"
                        f"<span style='font-weight:800;font-size:1.05em'>{_esc(title)}</span>"
                        f"<table style='width:100%;table-layout:fixed;border-collapse:collapse;margin:6px 0 0 0'>"
                        f"<colgroup><col style='width:65%'><col style='width:35%'></colgroup>"
                        f"<tr><th style='padding:4px 8px;text-align:left;font-weight:600'>Metric</th>"
                        f"<th style='padding:4px 8px;text-align:right;font-weight:600'>Value</th></tr>"
                        f"{rows}</table></div>"
                    )

                def _html_breakdown_table(title: str, rows_data: list[dict]) -> str:
                    """Build an HTML scoring breakdown table."""
                    _esc = html_mod.escape
                    header = "<tr><th style='padding:4px 8px;text-align:left;font-weight:600'>Component</th><th style='padding:4px 8px;text-align:right;font-weight:600'>Score</th><th style='padding:4px 8px;text-align:right;font-weight:600'>Weight</th></tr>"
                    rows = "".join(
                        f"<tr><td style='padding:4px 8px'>{_esc(str(r['Component']))}</td>"
                        f"<td style='padding:4px 8px;text-align:right'>{_esc(str(r['Score']))}</td>"
                        f"<td style='padding:4px 8px;text-align:right'>{_esc(str(r['Weight']))}</td></tr>"
                        for r in rows_data
                    )
                    return (
                        f"<div style='break-inside:avoid;margin-bottom:24px'>"
                        f"<span style='font-weight:800;font-size:1.05em'>{_esc(title)}</span>"
                        f"<table style='width:100%;table-layout:fixed;border-collapse:collapse;margin:6px 0 0 0'>"
                        f"<colgroup><col style='width:50%'><col style='width:25%'><col style='width:25%'></colgroup>"
                        f"{header}{rows}</table></div>"
                    )

                tbl_earnings = _html_table("Earnings & Price", {
                    "Trailing EPS": f"${data['trailing_eps']:.2f}" if data.get("trailing_eps") else "N/A",
                    "Trailing P/E": f"{data['trailing_pe']:.2f}" if data.get("trailing_pe") else "N/A",
                    "Forward P/E": f"{data['forward_pe']:.2f}" if data.get("forward_pe") else "N/A",
                    "P/S Ratio": f"{data['ps_ratio']:.2f}" if data.get("ps_ratio") else "N/A",
                    "Beta": f"{data['beta']:.2f}" if data.get("beta") else "N/A",
                })

                tbl_growth = _html_table("Growth", {
                    "0Y EPS Growth": f"{data['growth_current_year']:.1f}%" if data.get("growth_current_year") else "N/A",
                    "1Y EPS Growth": f"{data['growth_next_year']:.1f}%" if data.get("growth_next_year") else "N/A",
                    "5Y EPS Growth": f"{data['growth_5y']:.1f}%" if data.get("growth_5y") else "N/A",
                    "1Y Revenue Growth": f"{data['revenue_growth_next_year']:.1f}%" if data.get("revenue_growth_next_year") else "N/A",
                    "Past 3Y EPS Growth": f"{data['historical_growth_3y']:.1f}%" if data.get("historical_growth_3y") else "N/A",
                    "Past 5Y EPS Growth": f"{data['historical_growth_5y']:.1f}%" if data.get("historical_growth_5y") else "N/A",
                    "Past 3Y Sales Growth": f"{data['sales_growth_3y']:.1f}%" if data.get("sales_growth_3y") else "N/A",
                    "Past 5Y Sales Growth": f"{data['sales_growth_5y']:.1f}%" if data.get("sales_growth_5y") else "N/A",
                    "Eff. Growth (5Y damp.)": f"{scores['blended_growth']:.1f}%" if scores.get("blended_growth") else "N/A",
                })

                val_model_metrics = {
                    "PEG Ratio (ValueLens)": f"{scores['peg']:.2f}" if scores.get("peg") else "N/A",
                    "Fair P/E (ValueLens)": f"{peg_m['fair_pe']:.2f}" if peg_m.get("fair_pe") else "N/A",
                    "Median P/E (historical)": f"{hist_p['median_pe']:.2f}" if hist_p.get("median_pe") else "N/A",
                    "Hist. Fair P/E (ValueLens)": f"{hist_p['model_pe']:.2f}" if hist_p.get("model_pe") else "N/A",
                    "Historical Premium" + (" (disregarded)" if disregard_hist_premium else ""): f"{hist_p['premium']:.2f}x" if hist_p.get("premium") else "N/A",
                    "Margin of Safety": f"{round(valuation['margin_of_safety'] * 100)}%" if valuation.get("margin_of_safety") is not None else "N/A",
                    "Exit Premium": f"{round(valuation['exit_premium'] * 100)}%" if valuation.get("exit_premium") is not None else "N/A",
                }
                _nc = valuation.get("net_cash_per_share")
                if _nc is not None:
                    _nc_sign = "+" if _nc >= 0 else "-"
                    val_model_metrics["  Cash/Share"] = f"${data.get('cash_per_share', 0):,.2f}"
                    val_model_metrics["  Debt/Share"] = f"${data.get('debt_per_share', 0):,.2f}"
                    val_model_metrics["  Leases/Share"] = f"${data.get('capital_lease_per_share', 0):,.2f}"
                    val_model_metrics["Net Cash/Debt"] = f"{_nc_sign}${abs(_nc):,.2f}"
                tbl_valuation = _html_table("Valuation Model", val_model_metrics)

                breakdown_rows = []
                for key, display_label in {"peg": "PEG Ratio (ValueLens)", "eps_revisions": "EPS Revisions", "earnings_surprises": "Earnings Surprises"}.items():
                    bd = scores["breakdown"][key]
                    breakdown_rows.append({"Component": display_label, "Score": f"{bd['score']:.1f}/10", "Weight": f"{bd['weight']:.0%}"})
                tbl_scoring = _html_breakdown_table("Scoring Breakdown", breakdown_rows)

                st.markdown(
                    "<div style='display:flex;gap:1rem'>"
                    "<div style='flex:1 1 33%;min-width:0'>"
                    f"{tbl_valuation}"
                    "</div>"
                    "<div style='flex:1 1 33%;min-width:0'>"
                    f"{tbl_growth}"
                    "</div>"
                    "<div style='flex:1 1 33%;min-width:0'>"
                    f"{tbl_earnings}{tbl_scoring}"
                    "</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # Gate warnings
            warnings = [m for m in r["gate_messages"] if m.startswith("Warning")]
            if warnings:
                for w in warnings:
                    st.warning(w)

        except (KeyError, TypeError):
            # Stale or incompatible session state -- clear it silently
            del st.session_state["result"]

# ===== Batch Analysis tab =====
with tab_batch:
    st.subheader("Top 500 Global Stocks — Batch Analysis")
    st.caption("Only stocks passing all valuation gates are shown (positive EPS, available growth estimates).")

    if _CSV_PATH.exists():
        run_date = _read_batch_run_date()
        if run_date:
            st.caption(f"Last run: {run_date}")

    # --- Results table ---
    _batch_df = _load_batch_csv()
    if _batch_df is None:
        st.info("No batch results yet. Run batch_analyze.py from the command line.")
    else:
        # Drop legacy Status column if present
        if "Status" in _batch_df.columns:
            _batch_df = _batch_df[_batch_df["Status"] == "OK"].drop(columns=["Status"]).reset_index(drop=True)

        # Build deep-link URLs for each ticker
        _batch_df["Ticker"] = _batch_df["Ticker"].apply(
            lambda t: f"/?ticker={t}" if pd.notna(t) and t else t
        )

        display_cols = [
            "Ticker", "Name", "Score",
            "Market Cap", "Current Price",
            "Fair Value", "% vs Fair Value",
            "Fair Price", "% vs Fair Price",
        ]
        display_cols = [c for c in display_cols if c in _batch_df.columns]

        # Use # column as index so it renders narrow
        if "#" in _batch_df.columns:
            _batch_df = _batch_df.set_index("#")



        column_config = {
            "Ticker": st.column_config.LinkColumn(
                "Ticker",
                display_text=r"/\?ticker=(.+)",
            ),
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%d",
            ),
            "Current Price": st.column_config.NumberColumn(
                "Price", format="$%.2f",
            ),
            "Fair Value": st.column_config.NumberColumn(
                "Fair Value", format="$%.2f",
            ),
            "Fair Price": st.column_config.NumberColumn(
                "Fair Price", format="$%.2f",
            ),
            "% vs Fair Value": st.column_config.NumberColumn(
                "% vs FV", format="%.1f%%",
            ),
            "% vs Fair Price": st.column_config.NumberColumn(
                "% vs FP", format="%.1f%%",
            ),
        }

        styled = _batch_df[display_cols].style.apply(
            _apply_valuation_gradient, axis=None,
        )

        st.dataframe(
            styled,
            column_config=column_config,
            hide_index=False,
            height=600,
        )
