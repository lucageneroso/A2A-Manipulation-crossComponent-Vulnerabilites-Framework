"""
mas_report.py — MAS Campaign Report Generator
===============================================
[Pilastro 5 — Reporting e Visualizzazione]

Genera report completi (Markdown/HTML) a partire dai risultati 
delle campagne sperimentali MAS.

Il report include:
  1. Metriche globali (Total trials, pwnd rate, etc)
  2. EAPE-MAS Score unificato (Il rischio matematico)
  3. ARD Statistics (L'illusione della sicurezza - % di log falsi)
  4. Analisi Taint (Quanto e come il veleno si propaga)
  5. Suddivisione per Bias Cognitivo (Quale funziona meglio)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from framework.mas.ard_detector import ARDAggregateStats, ARDVerdict, ARDEvidence
from framework.mas.eape_mas import EAPEMASComputer, EAPEMASResult
from framework.mas.mas_runner import MASTrialResult
from framework.mas.smp import SMPComputer, SMPResult


class MASReportGenerator:
    """Generatore di report analitici per le campagne MAS."""

    def __init__(self, campaign_file: str):
        self.campaign_file = Path(campaign_file)
        self.data = self._load_data()

    def _load_data(self) -> dict:
        with open(self.campaign_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_markdown(self) -> str:
        """Genera un report in formato Markdown."""
        topology = self.data.get("topology", "Unknown")
        model = self.data.get("model", "Unknown")
        total_trials = self.data.get("total_trials", 0)
        successes = self.data.get("successes", 0)
        pwn_rate = successes / total_trials if total_trials > 0 else 0

        results = self.data.get("results", [])

        # Re-compute ARD Stats da dict
        ard_stats = ARDAggregateStats()
        for r in results:
            for role, verdict_str in r.get("ard_verdicts", {}).items():
                verdict = ARDVerdict(verdict_str)
                # confidence is approximated as 1.0 since it wasn't saved in simple format
                ard_stats.add_evidence(ARDEvidence(role, "write_file", verdict, 1.0))

        # Re-compute SMP (approx)
        trials_obj = []
        for r in results:
            trials_obj.append(
                MASTrialResult(
                    trial_id=r["trial_id"],
                    topology_name=r["topology_name"],
                    payload_name=r["payload_name"],
                    model_name=r["model_name"],
                    success=r["success"],
                    tool_call_executed=r.get("tool_call_executed", r["success"]),
                    researcher_compromised=r.get("researcher_compromised", False),
                )
            )

        smp_comp = SMPComputer()
        # Assumiamo W_taint_avg = 1.0 se non salvato nel json base
        smp_res = smp_comp.compute(trials_obj)

        eape_comp = EAPEMASComputer()
        eape_res = eape_comp.compute(
            p_exploit_l1=1.0, 
            p_exploit_l2=1.0, 
            smp_result=smp_res, 
            ard_stats=ard_stats
        )

        md = []
        md.append(f"# MAS Security Campaign Report: {topology}")
        md.append(f"**Model:** `{model}`")
        md.append(f"**Total Trials:** {total_trials}")
        md.append(f"**Successful Pwns:** {successes} ({pwn_rate:.1%})\n")

        md.append("## 1. Unified Risk Metric (EAPE-MAS)")
        md.append(f"**Score:** `{eape_res.eape_mas:.4f}` ({eape_res.risk_level})")
        md.append("- **P_exploit(L1):** 1.0 (Assumed fully compromised)")
        md.append(f"- **SMP Score:** {eape_res.smp_score:.2f} (Base SMP × Taint)")
        md.append("- **P_exploit(L2):** 1.0 (Assumed fail open if manipulated)")
        md.append(f"- **W_disconnect:** {eape_res.w_disconnect:.2f} (ARD Penalty)\n")

        md.append("## 2. Action-Reasoning Disconnect (ARD)")
        md.append("> *L'Illusione della Sicurezza: quanto spesso l'agente esegue il tool malevolo ma nei log dichiara di averlo rifiutato?*\n")
        
        md.append(f"- **Critical Disconnect Rate:** {ard_stats.critical_disconnect_rate:.1%} (Execute THEN Refuse)")
        md.append(f"- **False Security Rate:** {ard_stats.false_security_rate:.1%} (When attacked successfully, logs lied)")
        md.append(f"  - Clean Exploits (No disconnect): {ard_stats.consistent_execute}")
        md.append(f"  - Successful Defenses: {ard_stats.consistent_refuse}\n")

        md.append("## 3. Cognitive Bias Breakdown")
        md.append("| Bias | Intensity | Trials | Pwn Rate |")
        md.append("|------|-----------|--------|----------|")
        
        bias_stats = defaultdict(lambda: {"trials": 0, "pwns": 0})
        for r in results:
            p_name = r.get("payload_name", "unknown")
            parts = p_name.rsplit("_", 1)
            if len(parts) == 2:
                bias, intensity = parts
            else:
                bias, intensity = p_name, "unknown"
            
            key = f"{bias}_{intensity}"
            bias_stats[key]["trials"] += 1
            if r["success"]:
                bias_stats[key]["pwns"] += 1

        for key, stats in sorted(bias_stats.items()):
            t = stats["trials"]
            p = stats["pwns"]
            rate = p / t if t > 0 else 0
            bias, intensity = key.split("_", 1)
            md.append(f"| {bias} | {intensity} | {t} | {rate:.1%} |")

        return "\n".join(md)

    def save_markdown(self, output_path: str):
        md = self.generate_markdown()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
