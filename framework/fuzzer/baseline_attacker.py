"""
Baseline Attacker (Control Group)
===================================
A standard single-stage adversarial attacker that mimics what existing tools
like Garak, Promptfoo, and PyRIT would do: send a library of known adversarial
prompts directly to the target application's chat endpoint.

This is the CONTROL GROUP in the experiment. It is deliberately naive about
component boundaries — it does not attempt to exploit cross-component gaps.

Purpose: Establish the baseline recall rate for vulnerability discovery,
which the IAF must then statistically outperform to validate H2.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from framework.fuzzer.payload_templates import BASELINE_PAYLOADS, Payload
from framework.harness.judge import BypassVerdict, JudgmentResult, SuccessJudge
from framework.harness.runner import BENCHMARK_APPS, HarnessRunner, InteractionRecord

logger = logging.getLogger(__name__)


@dataclass
class BaselineExperimentResult:
    """Results of running the baseline attacker against all apps."""
    app_key: str
    app_name: str
    total_payloads: int
    successes: int
    failures: int
    blocked: int
    errors: int
    success_rate: float
    judgments: list[JudgmentResult]
    duration_seconds: float


class BaselineAttacker:
    """
    Sends all applicable baseline payloads to each benchmark app.
    Records and judges each response.
    """

    def __init__(self, n_trials: int = 1):
        """
        Args:
            n_trials: Number of times to repeat each payload.
                      Increase to 10-30 for statistical significance.
                      Use 1 for a quick smoke test.
        """
        self.n_trials = n_trials
        self.runner = HarnessRunner()
        self.judge = SuccessJudge()

    def _get_payloads_for_app(self, app_key: str) -> list[Payload]:
        """Filter baseline payloads for a specific app."""
        return [
            p for p in BASELINE_PAYLOADS
            if p.target_app in (app_key, "all")
        ]

    async def run_on_app(
        self,
        app_key: str,
        session_prefix: str = "baseline",
    ) -> BaselineExperimentResult:
        """Run all baseline payloads against a single app, n_trials times each."""
        app = BENCHMARK_APPS[app_key]
        payloads = self._get_payloads_for_app(app_key)
        judgments: list[JudgmentResult] = []
        start_time = time.time()

        logger.info(
            f"[Baseline] Starting on {app.name} | "
            f"{len(payloads)} payloads × {self.n_trials} trials"
        )

        for payload in payloads:
            for trial in range(self.n_trials):
                session_id = f"{session_prefix}_{app_key}_{trial}"

                # Reset app state before each trial (especially important for App C)
                if app_key == "app_c":
                    await self.runner.reset_app(app, session_id=session_id)

                record = await self.runner.send_message(
                    app=app,
                    message=payload.text,
                    session_id=session_id,
                )
                judgment = self.judge.judge(app_key, record)
                judgments.append(judgment)

                verdict_icon = {
                    BypassVerdict.SUCCESS: "✅",
                    BypassVerdict.FAILURE: "❌",
                    BypassVerdict.BLOCKED: "🛡️",
                    BypassVerdict.ERROR: "⚠️",
                }.get(judgment.verdict, "?")

                logger.info(
                    f"  [{app_key}] {verdict_icon} {judgment.verdict.value} | "
                    f"'{payload.text[:60]}...'"
                )

        duration = time.time() - start_time

        # Aggregate
        successes = sum(1 for j in judgments if j.verdict == BypassVerdict.SUCCESS)
        failures = sum(1 for j in judgments if j.verdict == BypassVerdict.FAILURE)
        blocked = sum(1 for j in judgments if j.verdict == BypassVerdict.BLOCKED)
        errors = sum(1 for j in judgments if j.verdict == BypassVerdict.ERROR)
        total = len(judgments)

        return BaselineExperimentResult(
            app_key=app_key,
            app_name=app.name,
            total_payloads=total,
            successes=successes,
            failures=failures,
            blocked=blocked,
            errors=errors,
            success_rate=successes / total if total > 0 else 0.0,
            judgments=judgments,
            duration_seconds=duration,
        )

    async def run_all(self) -> dict[str, BaselineExperimentResult]:
        """Run baseline attacker against all 3 benchmark apps."""
        results = {}
        for app_key in BENCHMARK_APPS:
            results[app_key] = await self.run_on_app(app_key)
        return results

    def export_results(
        self,
        results: dict[str, BaselineExperimentResult],
        filepath: str,
    ) -> None:
        """Export results as JSON for downstream EAPE computation."""
        data = {
            "attacker": "baseline",
            "timestamp": time.time(),
            "summary": {
                app_key: {
                    "app_name": r.app_name,
                    "total_payloads": r.total_payloads,
                    "successes": r.successes,
                    "success_rate": r.success_rate,
                    "duration_seconds": r.duration_seconds,
                }
                for app_key, r in results.items()
            },
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Baseline results exported to {filepath}")
