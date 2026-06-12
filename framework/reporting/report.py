"""
Report Generator
=================
Generates formatted experiment reports for the terminal and for export.
Uses Rich for beautiful console output.
"""

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

logger = logging.getLogger(__name__)
console = Console()


def print_eape_results(results_filepath: str) -> None:
    """Print a formatted EAPE comparison table from a JSON results file."""
    with open(results_filepath) as f:
        data = json.load(f)

    results = data.get("results", [])

    table = Table(
        title="EAPE Results — Expected Attack Path Exploitability",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("App", style="bold")
    table.add_column("Attack Path", style="dim")
    table.add_column("EAPE Score", justify="center")
    table.add_column("Risk", justify="center")
    table.add_column("Vulnerable?", justify="center")
    table.add_column("Min Trials", justify="right")
    table.add_column("Transitions P", style="dim")

    risk_colors = {
        "NONE": "green",
        "LOW": "yellow",
        "MEDIUM": "orange1",
        "HIGH": "red",
        "CRITICAL": "bold red",
    }

    for r in results:
        risk = r["risk_label"]
        color = risk_colors.get(risk, "white")
        eape_str = f"[{color}]{r['eape']:.4f}[/{color}]"
        risk_str = f"[{color}]{risk}[/{color}]"
        vuln_str = "[red]YES[/red]" if r["is_vulnerable"] else "[green]NO[/green]"
        probs_str = " × ".join(f"{p:.2f}" for p in r["transition_probabilities"])

        table.add_row(
            r["app_name"],
            r["path_name"],
            eape_str,
            risk_str,
            vuln_str,
            str(r["min_trials"]),
            probs_str,
        )

    console.print(table)


def print_comparison_table(baseline_file: str, iaf_file: str) -> None:
    """Print baseline vs. IAF comparison — the main H2 evidence table."""
    with open(baseline_file) as f:
        baseline = json.load(f)
    with open(iaf_file) as f:
        iaf = json.load(f)

    table = Table(
        title="Baseline Attacker vs. Interaction-Aware Fuzzer",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("App", style="bold")
    table.add_column("Baseline Successes", justify="center")
    table.add_column("Baseline Rate", justify="center")
    table.add_column("IAF Successes", justify="center")
    table.add_column("IAF Rate", justify="center")
    table.add_column("Δ Improvement", justify="center", style="bold green")

    baseline_summary = baseline.get("summary", {})
    iaf_summary = iaf.get("summary", {})

    for app_key in ["app_a", "app_b", "app_c"]:
        b = baseline_summary.get(app_key, {})
        i = iaf_summary.get(app_key, {})

        b_successes = b.get("successes", 0)
        b_rate = b.get("success_rate", 0.0)
        i_successes = i.get("successes", 0)
        i_rate = i.get("success_rate", 0.0)
        delta = i_rate - b_rate
        delta_str = f"+{delta:.1%}" if delta > 0 else f"{delta:.1%}"

        table.add_row(
            b.get("app_name", app_key),
            str(b_successes),
            f"{b_rate:.1%}",
            str(i_successes),
            f"{i_rate:.1%}",
            f"[green]{delta_str}[/green]" if delta > 0 else delta_str,
        )

    console.print(table)

    console.print(
        Panel(
            "[bold]Interpretation:[/bold]\n"
            "  • Baseline Attacker = standard prompt injection (Garak/Promptfoo approach)\n"
            "  • IAF = cross-component, boundary-aware attack patterns\n"
            "  • Δ Improvement = IAF advantage in discovering cross-component vulnerabilities\n\n"
            "[italic dim]Table directly corresponds to Hypothesis H2 validation.[/italic dim]",
            border_style="dim",
        )
    )


def print_attack_graph_summary(graph_summary: dict) -> None:
    """Print a formatted attack graph summary."""
    console.print(
        Panel(
            f"[bold]{graph_summary['app_name']}[/bold] — Attack Graph\n"
            f"Total paths: {graph_summary['total_paths']}\n"
            f"Max EAPE: [red]{graph_summary['max_eape']:.4f}[/red] (path: {graph_summary['max_eape_path']})",
            border_style="cyan",
        )
    )

    for path_data in graph_summary.get("paths", []):
        table = Table(
            title=f"Path: {path_data['path_name']}",
            box=box.SIMPLE,
            show_header=True,
        )
        table.add_column("Transition")
        table.add_column("P(T)", justify="center")
        table.add_column("Successes", justify="center")
        table.add_column("Trials", justify="center")

        for t in path_data["transitions"]:
            prob_color = "green" if t["probability"] < 0.1 else "red"
            table.add_row(
                f"{t['from']} → {t['to']}",
                f"[{prob_color}]{t['probability']:.3f}[/{prob_color}]",
                str(t["successes"]),
                str(t["trials"]),
            )

        bottleneck = path_data.get("bottleneck")
        console.print(table)
        if bottleneck:
            console.print(f"  [dim]Bottleneck: {bottleneck}[/dim]")

        eape = path_data.get("path_probability", 0.0)
        console.print(f"  EAPE = [bold]{eape:.4f}[/bold]\n")
