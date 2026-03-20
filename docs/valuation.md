# Valuation Model

The fair value estimate uses a single-flow approach: a model fair price adjusted by a historical premium, then applies beta-adjusted entry/exit price bands.

## Table of Contents

- [Valuation Flow](#valuation-flow)
- [Step 1: Fair Value (Compounding Model)](#step-1-fair-price-compounding-model)
- [Step 2: Historical Premium](#step-2-historical-premium)
- [Step 3: Net Cash Adjustment](#step-3-net-cash-adjustment)
- [Entry Price (Margin of Safety)](#entry-price-margin-of-safety)
- [Exit Price (Exit Premium)](#exit-price-exit-premium)
- [Currency/ADR Mismatch Detection](#currencyadr-mismatch-detection)

---

## Valuation Flow

```
fair_price = (Fair Value x Historical Premium) + Net Cash Adjustment
```

| Step | Component | Range | Default |
|------|-----------|-------|---------|
| 1 | Fair Value | model-derived | Required (None = no valuation) |
| 2 | Historical Premium | 0.80x - 1.20x | 1.0 (neutral when data insufficient) |
| 3 | Net Cash Adjustment | unclamped | 0 (when balance sheet data unavailable) |

Entry and exit prices are derived from the final fair value. The net cash adjustment is applied to all three prices (fair, entry, exit).

---

## Step 1: Fair Value (Compounding Model)

Derives fair price from the 5-year compounding model: pay today's no-growth P/E for the earnings the company will have in 5 years.

### Formula

```
Fair P/E = BASE_PE x (1 + effective_growth / 100)^5
Fair Value = Fair P/E x EPS
```

Where:
- **`BASE_PE = 12.0`**: P/E a zero-growth company deserves (~perpetuity at 7-8% discount rate)
- **`effective_growth`**: Dampened 5Y growth estimate (see [algorithms.md](algorithms.md#effective-growth-rate)), or dampened custom growth if `--growth` is used
- **`EPS`**: Trailing EPS, or custom EPS if `--eps` flag is used

### Examples

| EPS | Growth (raw) | Growth (dampened) | Fair P/E | Fair Value |
|-----|-------------|-------------------|----------|------------|
| $8.00 | 12% | 11.0% | 20.2 | $161.50 |
| $4.00 | 20% | 17.9% | 28.1 | $112.30 |
| $5.00 | -5% | -5.0% | 9.3 | $46.40 |
| $10.00 | 0% | 0% | 12.0 | $120.00 |

### Why This Model?

The traditional PEG=1 model (`Fair P/E = Growth%`) breaks down for low-growth stocks. A company growing at 10% would get a fair P/E of 10, yet the S&P 500 at ~10% growth has consistently traded above 20x. The compounding model fixes this by anchoring to a base perpetuity value (12x) and adding the compounded growth premium.

---

## Step 2: Historical Premium

Compares how the market actually valued the stock historically against what the model says it should have been valued at, given its historical growth rate. This captures persistent premiums (brand moat, quality perception) or discounts that the model alone cannot account for.

### Formula

```
Hist. Fair P/E = BASE_PE x (1 + historical_growth / 100)^5   (raw, NO dampening)
Median P/E     = median of 5Y historical trailing P/E series
Premium        = Median P/E / Hist. Fair P/E                  (clamped [0.80, 1.20])
```

### Key Design Decisions

- **Historical growth is NOT dampened**: It is backward-looking fact, not an optimistic analyst estimate
- **Uses time-appropriate EPS**: Each monthly price point is divided by the EPS that was current at that time (from annual income statements), not today's EPS
- **Clamped to [0.80, 1.20]**: Limits the historical signal to +-20% to prevent it from dominating the model

### Interpretation

| Premium | Meaning |
|---------|---------|
| 1.20 (cap) | Market consistently pays more than model predicts (strong moat) |
| 1.05 | Slight historical premium |
| 1.00 | Market matches model expectations |
| 0.95 | Slight historical discount |
| 0.80 (floor) | Market consistently pays less than model predicts |

### Fallback

When data is insufficient (no historical prices, no EPS, no historical growth), premium defaults to 1.0 (neutral -- no adjustment applied).

### Examples

| Stock | Median P/E | Hist. Growth | Hist. Fair P/E | Raw Premium | Clamped |
|-------|-----------|-------------|----------------|-------------|---------|
| AAPL | 33.3 | 6.1% | 16.1 | 2.07 | 1.20 |
| TSM | 26.9 | 21.0% | 31.2 | 0.86 | 0.86 |
| NVDA | 68.9 | 197.8% | 2812 | 0.02 | 0.80 |

Note: AAPL's premium is capped because its brand moat causes the market to consistently pay far above what the model predicts for ~6% growth. NVDA's historical growth is extreme (~198%), making the model's historical fair P/E unrealistically high, so the discount is floored.

---

## Step 3: Net Cash Adjustment

Shifts the earnings-based fair price from an enterprise-value-like estimate to an equity value by adding net cash per share. A company's equity is worth its earnings power plus any excess cash, minus any debt obligations.

### Formula

```
net_cash_per_share = cash_per_share - debt_per_share - capital_lease_per_share
fair_price         = (Fair Value x Historical Premium) + net_cash_per_share
entry_price        = entry_price + net_cash_per_share
exit_price         = exit_price + net_cash_per_share
```

### Components

| Field | Source | Notes |
|-------|--------|-------|
| Cash per share | `totalCash / sharesOutstanding` (yfinance) | Cash & short-term investments |
| Debt per share | `totalDebt / sharesOutstanding` (yfinance) | Short-term + long-term debt |
| Capital leases per share | `Capital Lease Obligations / sharesOutstanding` (balance sheet) | Defaults to 0 when absent |

All values are in `financialCurrency`. For ADRs/foreign listings where FX conversion is active, values are converted to price currency using the same `fx_rate` used for EPS.

### Why Applied After the Premium

The historical premium is an earnings quality multiplier — it scales how the market values this company's earnings stream. Cash and debt are balance sheet items: $1 of cash is worth $1 regardless of whether the market pays a 1.15x or 0.85x premium on earnings. Applying the adjustment after the premium avoids incorrectly scaling balance sheet items by an earnings-based multiplier.

### Why No Double-Counting

The PEG fair price is built from `BASE_PE x (1 + g)^5 x EPS` — a growth-compounding model anchored to a fixed base P/E, not the market's trailing P/E. Since the model's fair P/E is independent of how the market currently prices the stock's balance sheet, adding net cash does not double-count.

### Examples

| Stock | Cash/sh | Debt/sh | Leases/sh | Net Cash | Impact |
|-------|---------|---------|-----------|----------|--------|
| NVDA | $2.57 | $0.47 | $0.11 | +$2.00 | +1.1% |
| AAPL | $4.56 | $6.16 | $0.00 | -$1.61 | -0.8% |
| META | $37.30 | $38.90 | $11.50 | -$13.10 | -1.9% |
| ORCL | $6.88 | $45.83 | $4.01 | -$42.97 | -22.3% |

### Exclusions and Fallbacks

- **Financial sector**: Banks, insurance companies, and asset managers are excluded (`sector` = "Financial Services" or "Financial"). Their cash and debt are operating assets/liabilities, not excess cash or financing leverage.
- **Negative entry price gate**: If the net cash adjustment pushes the entry price below zero, a hard gate blocks the output. This signals that debt overwhelms earnings power and the model cannot produce a meaningful valuation.
- **Missing data**: When balance sheet data is unavailable (`totalCash`, `totalDebt`, or `sharesOutstanding` missing), no adjustment is applied (equivalent to net cash = 0).

---

## Entry Price (Growth Scenario)

The entry price represents the buy target: a pessimistic fair value computed by shocking growth downward, scaled by beta.

### Formula

```
shock         = 0.10 x (0.5 + 0.5 x beta)          clamped [0.05, 0.25]
entry_growth  = effective_growth x (1 - shock)       (worse direction for negative growth)
entry_price   = fair_pe(entry_growth) x EPS x historical_premium + net_cash_per_share
entry_discount clamped [0%, 10%]
```

### How It Works

The entry shock combines two risk factors:

1. **Growth uncertainty** — the compounding model amplifies growth errors over 5 years. A 20% miss on a 50% grower causes a ~29% error in fair value, but only ~3% for a 3% grower. The scenario approach naturally captures this.
2. **Price volatility (beta)** — high-beta stocks swing more, so you need a wider margin to actually buy the dip.

### Examples

| Stock | Beta | Shock | Growth | Entry Growth | Discount |
|-------|------|-------|--------|-------------|----------|
| PG | 0.34 | 6.7% | 3.6% | 3.4% | 3.0% (floor) |
| AAPL | 1.12 | 10.6% | 11.7% | 10.5% | 5.1% |
| META | 1.28 | 11.4% | 21.0% | 18.6% | 8.7% |
| NVDA | 2.38 | 16.9% | 51.6% | 42.9% | 15.0% (cap) |

### Design Rationale

The previous model used a flat beta-only margin (0-10%) which produced meaningless 1-2% margins for low-beta stocks and ignored the fact that high-growth fair values are inherently less certain. The scenario approach derives the margin from the model itself — it answers "how wrong is fair value if growth disappoints by this much?"

---

## Exit Price (P/E Stretch)

The exit price represents the sell target: a price above entry where the stock's valuation is historically exaggerated. The philosophy is buy-and-hold for the medium term, exiting only when the market overpays.

### Formula

```
pe_stretch   = P/E_90th / P/E_50th     (from 5Y trailing P/E distribution)
exit_premium = 0.25 + (pe_stretch - 1.0) x 0.50     clamped [0.25, 0.50]
exit_price   = entry_price x (1 + exit_premium) + net_cash_per_share
```

Fallback: `exit_premium = 0.30` when insufficient historical P/E data (< 10 data points).

### How It Works

The **P/E stretch** measures how much the market historically stretches this stock's valuation above its median. A stock with a tight, stable P/E band (like Visa or Coca-Cola) gets a lower exit premium — the market rarely "exaggerates" it. A stock with wide P/E swings (like Salesforce or Broadcom) gets a higher premium — the market does push it to extremes.

The premium is applied **above the entry price**, not above fair value. This reflects the investor's actual cost basis: "given where I bought, at what price is it time to sell?"

### Examples

| Stock | P/E 50th | P/E 90th | Stretch | Exit Premium |
|-------|---------|---------|---------|-------------|
| KO | 25.5 | 28.3 | 1.11 | 25% (floor) |
| AAPL | 33.3 | 38.6 | 1.16 | 33% |
| META | 27.9 | 37.8 | 1.35 | 43% |
| AMZN | 43.0 | 66.6 | 1.55 | 50% (cap) |

### Design Rationale

The previous model used a flat beta-adjusted premium (20-40% above fair value). This didn't account for how each stock's valuation actually behaves — a utility and a tech stock with the same beta got the same exit premium. The P/E stretch approach uses empirical market data specific to each stock: if the market has never pushed KO above 28x P/E, a 25% premium above entry is already generous. If AMZN regularly trades at 66x, you need a wider band to avoid selling prematurely.

---

## Currency/ADR Conversion

ADRs and foreign-listed stocks (e.g., TSM, BABA, SAP, NVO) report income statement EPS in their local `financialCurrency` (TWD, CNY, EUR, DKK) while trading prices and `trailingEps` are in the market `currency` (USD). This would produce nonsensical P/E ratios if left uncorrected.

### Automatic FX Conversion

When `currency != financialCurrency` in `stock.info`, the tool fetches a live FX rate and converts all income statement EPS to the price currency before any calculations:

```
1. Detect: info["currency"] != info["financialCurrency"]
2. Fetch:  FX rate via yf.Ticker("{financialCurrency}{currency}=X")
3. Convert: annual_eps_history[i].eps *= fx_rate
            quarterly_eps_history[i].eps *= fx_rate
```

This allows full historical P/E charts, forward P/E charts, and growth-adjusted valuations to work correctly for any foreign-listed stock.

A single current FX rate is used (not historical rates per time period). This is acceptable because:
- P/E ratios are scaled uniformly, preserving the median and relative shape
- Major FX pairs (TWD, EUR, DKK, CNY vs USD) have been relatively stable over 5-year windows
- The marginal error (~5%) is small compared to EPS estimation uncertainty

### Safety Net

If the FX fetch fails, a legacy 3x-divergence check detects the mismatch:

```
ratio = latest_annual_eps / trailing_eps_from_info
if ratio > 3.0 or ratio < 0.33:
    # Fall back to current trailing EPS for all historical P/E calculations
```

This fallback is less accurate (doesn't account for EPS growth over time) but avoids the much larger error of mixing currencies.
