from rich.console import Console
from rich.table import Table
from rich.text import Text


def print_output(
    ticker: str,
    data: dict,
    gate_messages: list[str],
    scores: dict,
    valuation: dict,
):
    """Print analysis results to terminal using rich."""
    console = Console(width=65)

    name = data.get("name", ticker)
    price = data.get("current_price")
    price_str = f"${price:,.2f}" if price else "N/A"

    # Header
    console.print()
    header = Text()
    header.append(f"{ticker}", style="bold white")
    header.append(f" - {name}", style="dim")
    header.append(f"  {price_str}", style="bold cyan")
    console.print(header)
    console.print("-" * 58, style="dim")

    _print_detailed(console, data, scores, valuation, gate_messages)

    console.print()


def print_gate_failure(ticker: str, name: str, messages: list[str]):
    """Print gate failure messages."""
    console = Console(width=65)
    console.print(f"\n[bold]{ticker} - {name}[/bold]")
    for msg in messages:
        console.print(f"  [red]{msg}[/red]")
    console.print()


def _print_concise(
    console: Console,
    data: dict,
    scores: dict,
    valuation: dict,
    warnings: list[str],
):
    score = scores["final_score"]
    label = scores["label"]
    color = _score_color(score)

    console.print(f"  Investibility Score:  [{color}]{score}/100  [{label}][/{color}]")

    if valuation["fair_value"]:
        console.print(f"  Fair Value:           ${valuation['fair_value']:,.2f}")
    if valuation["entry_price"]:
        margin_pct = round(valuation["margin_of_safety"] * 100)
        console.print(
            f"  Entry Price:          [green]<= ${valuation['entry_price']:,.2f}[/green]"
            f"  ({margin_pct}% margin of safety)"
        )
    if valuation["exit_price"]:
        exit_pct = round(valuation.get("exit_premium", 0) * 100)
        console.print(
            f"  Exit Price:           [red]>= ${valuation['exit_price']:,.2f}[/red]"
            f"  ({exit_pct}% above fair value)"
        )
    if scores["peg"] is not None:
        console.print(f"  PEG Ratio:            {scores['peg']:.2f}")

    _print_warnings(console, warnings)


def _print_detailed(
    console: Console,
    data: dict,
    scores: dict,
    valuation: dict,
    warnings: list[str],
):
    # Metrics table
    console.print("\n[bold]METRICS[/bold]")
    metrics_table = Table(show_header=False, box=None, padding=(0, 2))
    metrics_table.add_column("Metric", style="dim")
    metrics_table.add_column("Value")

    _add_metric_row(metrics_table, "Trailing P/E", data.get("trailing_pe"), fmt=".2f")
    _add_metric_row(metrics_table, "Forward P/E", data.get("forward_pe"), fmt=".2f")
    _add_metric_row(metrics_table, "Trailing EPS", data.get("trailing_eps"), prefix="$", fmt=".2f")
    _add_metric_row(metrics_table, "0Y EPS Growth", data.get("growth_current_year"), suffix="%", fmt=".1f")
    _add_metric_row(metrics_table, "+1Y EPS Growth", data.get("growth_next_year"), suffix="%", fmt=".1f")
    _add_metric_row(metrics_table, "5Y Est. Growth", data.get("growth_5y"), suffix="%", fmt=".1f")
    _add_metric_row(metrics_table, "1Y Revenue Growth", data.get("revenue_growth_next_year"), suffix="%", fmt=".1f")
    _add_metric_row(metrics_table, "Eff. Growth (5Y dampened)", scores.get("blended_growth"), suffix="%", fmt=".1f")
    if scores["peg"] is not None:
        metrics_table.add_row("PEG Ratio (ValueLens)", f"{scores['peg']:.2f}  (P/E / Fair P/E)")
    _add_metric_row(metrics_table, "P/S Ratio", data.get("ps_ratio"), fmt=".2f")
    if scores["psg"] is not None:
        metrics_table.add_row("PSG Ratio", f"{scores['psg']:.2f}")
    _add_metric_row(metrics_table, "Beta", data.get("beta"), fmt=".2f")

    console.print(metrics_table)

    # Scoring breakdown
    console.print("\n[bold]SCORING BREAKDOWN[/bold]")
    scoring_table = Table(show_header=True, box=None, padding=(0, 2))
    scoring_table.add_column("Metric", style="dim")
    scoring_table.add_column("Bar")
    scoring_table.add_column("Score", justify="right")
    scoring_table.add_column("Weight", justify="right", style="dim")

    for key, display_label in [
        ("peg", "PEG Ratio (ValueLens)"),
        ("psg", "PSG Ratio"),
        ("eps_revisions", "EPS Revisions"),
        ("earnings_surprises", "Earnings Surprises"),
    ]:
        bd = scores["breakdown"][key]
        bar = _make_bar(bd["score"])
        scoring_table.add_row(
            display_label,
            bar,
            f"{bd['score']:.1f}/10",
            f"{bd['weight']:.0%}",
        )

    console.print(scoring_table)

    score = scores["final_score"]
    label = scores["label"]
    color = _score_color(score)
    console.print(f"  {'-' * 50}")
    console.print(f"  Investibility Score:   [{color}]{score}/100  [{label}][/{color}]")

    # Valuation breakdown
    console.print("\n[bold]VALUATION[/bold]")
    val_table = Table(show_header=False, box=None, padding=(0, 2))
    val_table.add_column("Item", style="dim")
    val_table.add_column("Value")

    peg_m = valuation["peg_method"]
    hist_p = valuation["historical_premium"]

    if peg_m["fair_price"]:
        val_table.add_row(
            "Fair Price",
            f"${peg_m['fair_price']:,.2f}  (fair P/E={peg_m['fair_pe']})"
        )
    if hist_p["median_pe"]:
        premium_str = f"{hist_p['premium']:.2f}x"
        details = f"median P/E={hist_p['median_pe']}"
        if hist_p.get("model_pe"):
            details += f"\n              Hist. Fair P/E={hist_p['model_pe']}"
        val_table.add_row(
            "Historical Premium",
            f"{premium_str}  ({details})"
        )
    if valuation["fair_value"]:
        val_table.add_row(
            "Fair Value",
            f"[bold]${valuation['fair_value']:,.2f}[/bold]"
        )
    margin_pct = round(valuation["margin_of_safety"] * 100)
    exit_pct = round(valuation.get("exit_premium", 0) * 100)
    val_table.add_row("Margin of Safety", f"{margin_pct}% (beta-adjusted)")
    val_table.add_row("Exit Premium", f"{exit_pct}% (beta-adjusted)")
    if valuation["entry_price"]:
        val_table.add_row("Entry Price", f"[green]<= ${valuation['entry_price']:,.2f}[/green]")
    if valuation["exit_price"]:
        val_table.add_row("Exit Price", f"[red]>= ${valuation['exit_price']:,.2f}[/red]")

    console.print(val_table)

    _print_warnings(console, warnings)


def _print_warnings(console: Console, warnings: list[str]):
    warning_msgs = [w for w in warnings if w.startswith("Warning")]
    if warning_msgs:
        console.print("\n[bold yellow]WARNINGS[/bold yellow]")
        for w in warning_msgs:
            console.print(f"  [yellow]{w}[/yellow]")


def _make_bar(score: float, width: int = 10) -> str:
    filled = round(score / 10 * width)
    empty = width - filled
    return f"[green]{'#' * filled}[/green][dim]{'.' * empty}[/dim]"


def _score_color(score: int) -> str:
    if score >= 80:
        return "bold green"
    if score >= 70:
        return "green"
    if score >= 50:
        return "yellow"
    if score >= 20:
        return "red"
    return "bold red"


def _add_metric_row(table: Table, label: str, value, prefix: str = "", suffix: str = "", fmt: str = ""):
    if value is None:
        table.add_row(label, "N/A")
    else:
        formatted = f"{prefix}{value:{fmt}}{suffix}" if fmt else f"{prefix}{value}{suffix}"
        table.add_row(label, formatted)
