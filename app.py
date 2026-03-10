import base64
import statistics
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_fetcher import fetch_stock_data
from src.gates import check_gates
from src.scorer import apply_price_cap, score_stock
from src.valuator import (
    calculate_valuation,
    compute_historical_forward_pe_series,
    compute_historical_pe_series,
)

_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(ticker: str, cache_version: int = 0) -> dict:
    """Cache stock data for 1 hour.

    cache_version is incremented when finviz failed on a previous fetch,
    so the next submit bypasses the stale cache and retries finviz.
    """
    return fetch_stock_data(ticker)


# --- Page config ---
st.set_page_config(page_title="ValueLens", page_icon=str(_LOGO_PATH), layout="centered")
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


# --- Section 1: Input bar ---
with st.form("analyze_form"):
    col_ticker, col_button = st.columns([3, 1])
    with col_ticker:
        ticker = st.text_input("Ticker", placeholder="e.g. NVDA", label_visibility="collapsed")
    with col_button:
        analyze = st.form_submit_button("Analyze", type="primary", width="stretch")

with st.expander("Advanced Options"):
    use_custom_eps = st.checkbox(
        "Override trailing EPS",
        help="Override trailing EPS for stocks with GAAP distortions.",
    )
    custom_eps = None
    if use_custom_eps:
        custom_eps = st.number_input(
            "Custom EPS",
            min_value=0.01,
            value=1.00,
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
            value=15.0,
            step=0.5,
            format="%.1f",
            placeholder="e.g. 15.0",
        )
    disregard_hist_premium = st.checkbox(
        "Disregard Historical Premium",
        help="Still calculated and shown, but not applied to Fair Value.",
    )


# --- Orchestration ---
if analyze and ticker:
    ticker = ticker.upper().strip()

    fetch_error = None
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            # Bust cache for tickers where finviz previously failed
            finviz_fails = st.session_state.setdefault("_finviz_fails", {})
            cache_ver = finviz_fails.get(ticker, 0)
            data = cached_fetch(ticker, cache_version=cache_ver)

            if not data.get("_finviz_ok", True):
                # Next submit will use a new cache key → retry finviz
                finviz_fails[ticker] = cache_ver + 1
            else:
                finviz_fails.pop(ticker, None)
        except Exception as e:
            fetch_error = e

    if fetch_error is not None:
        st.error(f"Error fetching data for {ticker}: {fetch_error}")
        st.stop()

    passed, gate_messages = check_gates(data, custom_growth=custom_growth)

    if not passed:
        st.error(f"**{ticker} — {data.get('name', 'Unknown')}**")
        for msg in gate_messages:
            st.error(msg)
        st.stop()

    scores = score_stock(data, custom_eps=custom_eps, custom_growth=custom_growth)
    valuation = calculate_valuation(
        data, custom_eps=custom_eps, custom_growth=custom_growth, scores=scores,
        disregard_hist_premium=disregard_hist_premium,
    )
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
            logo_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"

        if logo_url:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px'>"
                f"<img src='{logo_url}' width='40' height='40' style='border-radius:6px'"
                f" onerror=\"this.style.display='none'\">"
                f"<span style='font-size:1.5em;font-weight:600'>{ticker} — {name}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"### {ticker} — {name}")
        if price:
            st.markdown(f"**Current Price:** ${price:,.2f}")

        score = scores["final_score"]
        label = scores["label"]

        score_colors = {
            "Strong Buy": "green", "Attractive": "green",
            "Hold / Fair Value": "orange", "Unattractive": "red", "Avoid": "red",
        }
        score_color = score_colors.get(label, "gray")
        st.markdown(
            f"<h2 style='color:{score_color};margin-top:0'>{score}/100 — {label}</h2>",
            unsafe_allow_html=True,
        )

        # Metric cards
        col1, col2, col3 = st.columns(3)
        with col1:
            fv = valuation.get("fair_value")
            st.metric("Fair Value", f"${fv:,.2f}" if fv else "N/A")
        with col2:
            ep = valuation.get("entry_price")
            margin_pct = round(valuation.get("margin_of_safety", 0) * 100)
            st.metric(
                "Entry Price",
                f"<= ${ep:,.2f}" if ep else "N/A",
                delta=f"{margin_pct}% margin",
                delta_color="off",
            )
        with col3:
            xp = valuation.get("exit_price")
            exit_pct = round(valuation.get("exit_premium", 0) * 100)
            st.metric(
                "Exit Price",
                f">= ${xp:,.2f}" if xp else "N/A",
                delta=f"{exit_pct}% above entry",
                delta_color="off",
            )

        if scores.get("peg") is not None:
            st.markdown(f"**PEG Ratio (ValueLens):** {scores['peg']:.2f}")

        st.divider()

        # --- Section 3: Price chart ---
        historical_prices = data.get("historical_prices")
        if historical_prices:
            dates = [p["date"] for p in historical_prices]
            prices = [p["price"] for p in historical_prices]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=prices, mode="lines",
                name="Price", line=dict(color="#2196F3", width=2),
            ))

            fv = valuation.get("fair_value")
            ep = valuation.get("entry_price")
            xp = valuation.get("exit_price")

            if ep:
                fig.add_hline(
                    y=ep, line_dash="dash", line_color="green",
                    annotation_text=f"Entry ${ep:,.0f}",
                    annotation_position="bottom right",
                )
            if fv:
                fig.add_hline(
                    y=fv, line_dash="solid", line_color="blue",
                    annotation_text=f"Fair Value ${fv:,.0f}",
                    annotation_position="top right",
                )
            if xp:
                fig.add_hline(
                    y=xp, line_dash="dash", line_color="red",
                    annotation_text=f"Exit ${xp:,.0f}",
                    annotation_position="top right",
                )

            fig.update_layout(
                title=f"{ticker} — Price vs Fair Value",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                showlegend=False,
            )

            st.plotly_chart(fig, width="stretch")

        # --- Section 4: Historical P/E chart ---
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
                title=f"{ticker} — Historical P/E",
                xaxis_title="Date",
                yaxis_title="P/E Ratio",
                height=350,
                showlegend=False,
            )

            st.plotly_chart(fig_pe, width="stretch")
        else:
            st.info("Historical P/E chart unavailable — insufficient positive earnings history.")

        # --- Section 4b: Historical Forward P/E chart ---
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
                title=f"{ticker} — Historical Forward P/E",
                xaxis_title="Date",
                yaxis_title="Forward P/E",
                height=350,
                showlegend=False,
            )

            st.plotly_chart(fig_fwd, width="stretch")
        else:
            st.info("Historical Forward P/E chart unavailable — insufficient earnings data.")

        # --- Section 5: Detailed Metrics (expandable) ---
        with st.expander("Detailed Metrics"):

            # Force consistent column widths across metric tables
            st.markdown(
                "<style>"
                "[data-testid='stExpander'] table { table-layout: fixed; width: 100%; }"
                "[data-testid='stExpander'] table th:first-child,"
                "[data-testid='stExpander'] table td:first-child { width: 65%; }"
                "table.val-breakdown { table-layout: fixed; width: 100%; }"
                "table.val-breakdown th:first-child,"
                "table.val-breakdown td:first-child { width: 40%; }"
                "table.val-breakdown th:nth-child(2),"
                "table.val-breakdown td:nth-child(2) { width: 25%; }"
                "table.val-breakdown th:nth-child(3),"
                "table.val-breakdown td:nth-child(3) { width: 35%; }"
                "</style>",
                unsafe_allow_html=True,
            )

            peg_m = valuation.get("peg_method", {})
            hist_p = valuation.get("historical_premium", {})

            def _metrics_table(metrics: dict) -> None:
                df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]).set_index("Metric")
                st.table(df)

            st.markdown("**Earnings & Price**")
            _metrics_table({
                "Trailing EPS": f"${data['trailing_eps']:.2f}" if data.get("trailing_eps") else "N/A",
                "Trailing P/E": f"{data['trailing_pe']:.2f}" if data.get("trailing_pe") else "N/A",
                "Forward P/E": f"{data['forward_pe']:.2f}" if data.get("forward_pe") else "N/A",
                "P/S Ratio": f"{data['ps_ratio']:.2f}" if data.get("ps_ratio") else "N/A",
                "Beta": f"{data['beta']:.2f}" if data.get("beta") else "N/A",
            })

            growth_sources = sorted({
                src for key in (
                    "growth_current_year_source", "growth_next_year_source",
                    "growth_5y_source", "historical_growth_5y_source",
                )
                if (src := data.get(key)) and src != "N/A"
            } | ({"finviz"} if any(
                data.get(k) is not None
                for k in ("historical_growth_3y", "sales_growth_3y", "sales_growth_5y")
            ) else set()))
            growth_src_suffix = f" ({', '.join(growth_sources)})" if growth_sources else ""
            st.markdown(f"**Growth{growth_src_suffix}**")

            _metrics_table({
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

            st.markdown("**Valuation Model**")
            _metrics_table({
                "PEG Ratio (ValueLens)": f"{scores['peg']:.2f}" if scores.get("peg") else "N/A",
                "Fair P/E (ValueLens)": f"{peg_m['fair_pe']:.2f}" if peg_m.get("fair_pe") else "N/A",
                "Median P/E (historical)": f"{hist_p['median_pe']:.2f}" if hist_p.get("median_pe") else "N/A",
                "Hist. Fair P/E (ValueLens)": f"{hist_p['model_pe']:.2f}" if hist_p.get("model_pe") else "N/A",
                "Historical Premium" + (" (disregarded)" if disregard_hist_premium else ""): f"{hist_p['premium']:.2f}x" if hist_p.get("premium") else "N/A",
                "Margin of Safety": f"{round(valuation['margin_of_safety'] * 100)}%" if valuation.get("margin_of_safety") is not None else "N/A",
                "Exit Premium": f"{round(valuation['exit_premium'] * 100)}%" if valuation.get("exit_premium") is not None else "N/A",
            })

            st.markdown("**Scoring Breakdown**")
            breakdown_labels = {
                "peg": "PEG Ratio (ValueLens)",
                "eps_revisions": "EPS Revisions",
                "earnings_surprises": "Earnings Surprises",
            }
            breakdown_rows = []
            for key, display_label in breakdown_labels.items():
                bd = scores["breakdown"][key]
                breakdown_rows.append({
                    "Component": display_label,
                    "Score": f"{bd['score']:.1f}/10",
                    "Weight": f"{bd['weight']:.0%}",
                })
            breakdown_df = pd.DataFrame(breakdown_rows).set_index("Component")
            st.table(breakdown_df)

            st.markdown("**Valuation Breakdown**")
            peg_mv = valuation["peg_method"]
            hist_pv = valuation["historical_premium"]
            margin_pct = round(valuation.get("margin_of_safety", 0) * 100)
            exit_pct = round(valuation.get("exit_premium", 0) * 100)
            val_rows = []

            if peg_mv.get("fair_price"):
                val_rows.append({
                    "Method": "Fair Price",
                    "Value": f"${peg_mv['fair_price']:,.2f}",
                    "Detail": f"Fair P/E = {peg_mv['fair_pe']}",
                })

            premium = hist_pv.get("premium", 1.0)
            if hist_pv.get("median_pe"):
                detail = f"Median P/E = {hist_pv['median_pe']}"
                if hist_pv.get("model_pe"):
                    detail += f"<br>Hist. Fair P/E = {hist_pv['model_pe']}"
                hp_label = "Historical Premium"
                if disregard_hist_premium:
                    hp_label += " <i>(disregarded)</i>"
                val_rows.append({
                    "Method": hp_label,
                    "Value": f"{premium:.2f}x",
                    "Detail": detail,
                })

            if valuation.get("fair_value"):
                val_rows.append({
                    "Method": "<b>Fair Value</b>",
                    "Value": f"<b>${valuation['fair_value']:,.2f}</b>",
                    "Detail": "",
                })

            if val_rows:
                html = "<table class='val-breakdown'>"
                html += "<thead><tr><th>Method</th><th>Value</th><th>Detail</th></tr></thead><tbody>"
                for r_ in val_rows:
                    html += f"<tr><td>{r_['Method']}</td><td>{r_['Value']}</td><td>{r_['Detail']}</td></tr>"
                html += "</tbody></table>"
                st.markdown(html, unsafe_allow_html=True)

        # Gate warnings
        warnings = [m for m in r["gate_messages"] if m.startswith("Warning")]
        if warnings:
            for w in warnings:
                st.warning(w)

    except (KeyError, TypeError):
        # Stale or incompatible session state — clear it silently
        del st.session_state["result"]
