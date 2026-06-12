"""
Experiment: Interaction-Aware Fuzzer Run
==========================================
Runs the IAF against all 3 benchmark apps, computes EAPE scores,
and saves results for comparison with the baseline.

Usage:
    python experiments/run_iaf.py [--trials N]
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.panel import Panel

from framework.fuzzer.interaction_aware_fuzzer import InteractionAwareFuzzer
from framework.harness.runner import HarnessRunner
from framework.metric.eape import EAPEComputer

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

app = typer.Typer()


@app.command()
def main(
    trials: int = typer.Option(10, "--trials", "-n", help="Number of trials per attack pattern (30+ for publication)"),
    output_dir: str = typer.Option("experiments/results", "--output", "-o", help="Output directory for results"),
    pattern: str = typer.Option("all", "--pattern", "-p", help="Attack pattern to run: all | app_a | app_b | app_c"),
):
    """Run the Interaction-Aware Fuzzer against benchmark apps and compute EAPE."""
    asyncio.run(_run(trials, output_dir, pattern))


async def _run(trials: int, output_dir: str, pattern: str):
    console.print(Panel(
        "[bold red]PenTesLLM — Interaction-Aware Fuzzer[/bold red]\n"
        f"Trials: {trials} | Pattern: {pattern}\n"
        "Cross-component attack patterns targeting component boundary gaps.\n"
        "[dim italic]EAPE metric will be computed after each pattern.[/dim italic]",
        border_style="red",
    ))

    # 1. Health check
    runner = HarnessRunner()
    console.print("\n[bold]Checking app health...[/bold]")
    health = await runner.check_all_healthy()
    if not all(health.values()):
        unhealthy = [k for k, v in health.items() if not v]
        console.print(f"[red]Apps not healthy: {unhealthy}[/red]")
        console.print("[dim]Start with: docker-compose up -d[/dim]")
        raise typer.Exit(1)

    # 2. Run IAF
    fuzzer = InteractionAwareFuzzer(n_trials=trials)
    start = time.time()

    if pattern == "all":
        results = await fuzzer.run_all()
    elif pattern == "app_a":
        results = {"app_a": await fuzzer.run_pattern_1_semantic_probe(trials)}
    elif pattern == "app_b":
        results = {"app_b": await fuzzer.run_pattern_2_split_context(trials)}
    elif pattern == "app_c":
        results = {"app_c": await fuzzer.run_pattern_3_history_flood(trials)}
    else:
        console.print(f"[red]Unknown pattern: {pattern}[/red]")
        raise typer.Exit(1)

    duration = time.time() - start

    # 3. Compute EAPE for each result
    eape_computer = EAPEComputer(n_trials_recommended=trials)
    eape_results = []

    for app_key, iaf_result in results.items():
        graph = iaf_result.attack_graph
        for path_name in graph.paths:
            try:
                eape_result = eape_computer.compute(graph, path_name)
                eape_results.append(eape_result)
            except Exception as e:
                console.print(f"[yellow]EAPE computation failed for {app_key}/{path_name}: {e}[/yellow]")

    # 4. Print results
    console.print(f"\n[bold]IAF complete in {duration:.1f}s[/bold]\n")

    from rich.table import Table, box
    table = Table(
        title="IAF Results with EAPE",
        box=box.ROUNDED,
        header_style="bold red",
    )
    table.add_column("App")
    table.add_column("Pattern")
    table.add_column("Attempts", justify="center")
    table.add_column("Successes", justify="center")
    table.add_column("Success Rate", justify="center")
    table.add_column("EAPE", justify="center")
    table.add_column("Risk", justify="center")

    risk_colors = {"NONE": "green", "LOW": "yellow", "MEDIUM": "orange1", "HIGH": "red", "CRITICAL": "bold red"}

    for app_key, iaf_result in results.items():
        # Find matching EAPE result
        eape = next((e for e in eape_results if e.app_name == iaf_result.app_name), None)
        eape_val = f"{eape.eape:.4f}" if eape else "N/A"
        risk = eape.risk_label if eape else "N/A"
        color = risk_colors.get(risk, "white")

        table.add_row(
            iaf_result.app_name,
            iaf_result.attack_pattern,
            str(iaf_result.total_attempts),
            str(iaf_result.successes),
            f"{iaf_result.success_rate:.1%}",
            f"[{color}]{eape_val}[/{color}]",
            f"[{color}]{risk}[/{color}]",
        )

    console.print(table)

    if eape_results:
        console.print(
            Panel(
                "[bold]EAPE Formula:[/bold] EAPE = PRODUCT P(T_i -> T_{i+1})\n"
                "Each P(T) is the empirically estimated probability of crossing a component boundary.\n"
                "[dim]A non-zero EAPE confirms a cross-component attack path exists.[/dim]",
                border_style="dim",
            )
        )

    # 5. Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())

    iaf_filepath = output_path / f"iaf_{timestamp}.json"
    fuzzer.export_results(results, str(iaf_filepath))

    if eape_results:
        eape_filepath = output_path / f"eape_{timestamp}.json"
        eape_computer.export_results(eape_results, str(eape_filepath))
        console.print(f"[green]EAPE results saved to {eape_filepath}[/green]")

    console.print(f"[green]IAF results saved to {iaf_filepath}[/green]")


if __name__ == "__main__":
    app()
