"""
Experiment: Baseline Attacker Run
===================================
Runs the baseline attacker (control group) against all 3 benchmark apps
and saves results for EAPE computation and comparison.

Usage:
    python experiments/run_baseline.py [--trials N]
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.panel import Panel

from framework.fuzzer.baseline_attacker import BaselineAttacker
from framework.harness.runner import BENCHMARK_APPS, HarnessRunner

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

app = typer.Typer()


@app.command()
def main(
    trials: int = typer.Option(1, "--trials", "-n", help="Number of trials per payload (30+ for publication)"),
    output_dir: str = typer.Option("experiments/results", "--output", "-o", help="Output directory for results"),
):
    """Run the baseline attacker (control group) against all 3 benchmark apps."""
    asyncio.run(_run(trials, output_dir))


async def _run(trials: int, output_dir: str):
    console.print(Panel(
        "[bold]PenTesLLM — Baseline Attacker[/bold]\n"
        f"Trials per payload: {trials}\n"
        "This is the control group — standard single-stage prompt injection.\n"
        "[dim italic]Results will be compared against the IAF.[/dim italic]",
        border_style="blue",
    ))

    # 1. Health check
    runner = HarnessRunner()
    console.print("\n[bold]Checking app health...[/bold]")
    health = await runner.check_all_healthy()
    if not all(health.values()):
        unhealthy = [k for k, v in health.items() if not v]
        console.print(f"[red]ERROR: Apps not healthy: {unhealthy}[/red]")
        console.print("[dim]Make sure docker-compose is running: docker-compose up -d[/dim]")
        raise typer.Exit(1)

    console.print("[green]All apps healthy. Starting experiment...[/green]\n")

    # 2. Run baseline
    attacker = BaselineAttacker(n_trials=trials)
    start = time.time()
    results = await attacker.run_all()
    duration = time.time() - start

    # 3. Summary
    console.print(f"\n[bold]Experiment complete in {duration:.1f}s[/bold]\n")

    from rich.table import Table, box
    table = Table(title="Baseline Attacker Summary", box=box.ROUNDED, header_style="bold blue")
    table.add_column("App")
    table.add_column("Payloads", justify="center")
    table.add_column("Successes", justify="center")
    table.add_column("Blocked", justify="center")
    table.add_column("Success Rate", justify="center")

    for app_key, result in results.items():
        rate_color = "green" if result.success_rate < 0.1 else "red"
        table.add_row(
            result.app_name,
            str(result.total_payloads),
            str(result.successes),
            str(result.blocked),
            f"[{rate_color}]{result.success_rate:.1%}[/{rate_color}]",
        )
    console.print(table)

    # 4. Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    filepath = output_path / f"baseline_{timestamp}.json"
    attacker.export_results(results, str(filepath))
    console.print(f"\n[green]Results saved to {filepath}[/green]")


if __name__ == "__main__":
    app()
