"""
run_mas_campaign.py — MAS Ablation Study Campaign Runner
==========================================================
[Pilastro 5 — Ablation Study & Reporting]

Script per avviare la campagna di validazione sperimentale sulle
vulnerabilità MAS (A2AM e Action-Reasoning Disconnect).

Esegue N trials su ogni topologia con i payload prioritari e
genera il report finale.

Usage:
  python experiments/run_mas_campaign.py
"""

import argparse
import logging
from pathlib import Path

from framework.mas.mas_topology import build_topology
from framework.mas.a2am_payloads import build_priority_payloads, build_adaptive_payload, build_rlhf_bypass_payload
from framework.mas.mas_runner import MASRunner
from framework.mas.mas_report import MASReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.add_argument_group("MAS Campaign")
    parser.add_argument(
        "--output-dir", type=str, default="experiments/results/mas",
        help="Directory per i risultati",
    )
    parser.add_argument(
        "--model", type=str, default="ollama/llama3.1:8b",
        help="Modello da testare",
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Numero di trials per payload",
    )
    parser.add_argument(
        "--topologies", type=str, nargs="+", default=["CHAIN_2", "CHAIN_3"],
        help="Topologie da testare",
    )
    parser.add_argument(
        "--adaptive", action="store_true",
        help="Abilita il payload adattivo multi-turn",
    )
    parser.add_argument(
        "--defense-level", type=str, default="standard",
        choices=["weak", "standard", "strong"],
        help="Livello di difesa dell'agente target",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.campaign_type == "rlhf_bypass":
        payloads = [build_rlhf_bypass_payload()]
        args.adaptive = True  # RLHF Bypass is always adaptive
    elif args.adaptive:
        payloads = [build_adaptive_payload()]
    else:
        payloads = build_priority_payloads()
        
    logger.info(f"Avvio campagna MAS: {len(payloads)} payloads, {args.trials} trials ciascuno. Adaptive: {args.adaptive}, Type: {args.campaign_type}")

    runner = MASRunner(output_dir=str(output_dir), model=args.model)

    for top_name in args.topologies:
        logger.info(f"\n{'='*50}\nEsecuzione Topologia: {top_name}\n{'='*50}")
        try:
            topology = build_topology(top_name, defense_level=args.defense_level)
        except ValueError as e:
            logger.error(f"Skipping {top_name}: {e}")
            continue

        # Esegui campagna
        results = runner.run_campaign(topology, payloads, n_trials=args.trials, is_adaptive=args.adaptive)

        if results:
            # Trova l'ultimo json salvato
            json_files = list(output_dir.glob(f"campaign_{top_name}_*.json"))
            if json_files:
                latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
                
                # Genera report markdown
                report_gen = MASReportGenerator(str(latest_json))
                report_path = output_dir / f"report_{top_name}.md"
                report_gen.save_markdown(str(report_path))
                logger.info(f"Report Markdown generato in: {report_path}")

    logger.info("Campagna globale completata.")

if __name__ == "__main__":
    import sys
    # workaround for argparse in script without importing it completely if not main
    parser = argparse.ArgumentParser(description="MAS Campaign Runner")
    parser.add_argument(
        "--output-dir", type=str, default="experiments/results/mas",
        help="Directory per i risultati",
    )
    parser.add_argument(
        "--model", type=str, default="ollama/llama3.1:8b",
        help="Modello da testare",
    )
    parser.add_argument(
        "--trials", type=int, default=5,
        help="Numero di trials per payload (default 5 per test rapido)",
    )
    parser.add_argument(
        "--topologies", type=str, nargs="+", default=["CHAIN_2"],
        help="Topologie da testare",
    )
    parser.add_argument(
        "--adaptive", action="store_true",
        help="Abilita il payload adattivo multi-turn",
    )
    parser.add_argument(
        "--defense-level", type=str, default="standard",
        choices=["weak", "standard", "strong"],
        help="Livello di difesa dell'agente target",
    )
    parser.add_argument(
        "--campaign-type", type=str, default="standard",
        choices=["standard", "rlhf_bypass"],
        help="Tipo di campagna",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.campaign_type == "rlhf_bypass":
        payloads = [build_rlhf_bypass_payload()]
        args.adaptive = True  # RLHF Bypass is always adaptive
    elif args.adaptive:
        payloads = [build_adaptive_payload()]
    else:
        payloads = build_priority_payloads()
        
    logger.info(f"Avvio campagna MAS su {args.model}: {len(payloads)} payloads, {args.trials} trials. Adaptive: {args.adaptive}, Type: {args.campaign_type}")

    runner = MASRunner(output_dir=str(output_dir), model=args.model)

    for top_name in args.topologies:
        logger.info(f"\n{'='*50}\nEsecuzione Topologia: {top_name}\n{'='*50}")
        try:
            topology = build_topology(top_name, defense_level=args.defense_level, campaign_type=args.campaign_type)
        except ValueError as e:
            logger.error(f"Skipping {top_name}: {e}")
            continue

        results = runner.run_campaign(topology, payloads, n_trials=args.trials, is_adaptive=args.adaptive)

        if results:
            json_files = list(output_dir.glob(f"campaign_{top_name}_*.json"))
            if json_files:
                latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
                report_gen = MASReportGenerator(str(latest_json))
                report_path = output_dir / f"report_{top_name}.md"
                report_gen.save_markdown(str(report_path))
                logger.info(f"Report Markdown generato: {report_path}")
