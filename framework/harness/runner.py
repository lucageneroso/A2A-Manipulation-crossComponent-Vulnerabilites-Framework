"""
Harness: Test Runner
=====================
Sends HTTP requests to benchmark apps and collects raw responses.
Handles retries, timeouts, and structured logging of all interactions.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class TargetApp:
    """Defines a benchmark application endpoint."""
    name: str
    base_url: str
    chat_endpoint: str = "/chat"
    reset_endpoint: str = "/reset"
    health_endpoint: str = "/health"


@dataclass
class InteractionRecord:
    """A single request-response pair with metadata."""
    app_name: str
    payload: str
    response: dict[str, Any]
    status_code: int
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    error: str | None = None


# ── Default App Registry ──────────────────────────────────────────────────────
BENCHMARK_APPS = {
    "app_a": TargetApp(
        name="App-A-Semantic-Bypass",
        base_url="http://localhost:8001",
    ),
    "app_b": TargetApp(
        name="App-B-RAG-Tool-Poisoning",
        base_url="http://localhost:8002",
    ),
    "app_c": TargetApp(
        name="App-C-Context-Truncation",
        base_url="http://localhost:8003",
    ),
}


class HarnessRunner:
    """
    Sends adversarial payloads to benchmark apps and records structured results.
    Supports both single requests and batch runs.
    """

    def __init__(
        self,
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self.timeout = timeout
        self.max_retries = max_retries

    async def send_message(
        self,
        app: TargetApp,
        message: str,
        session_id: str = "experiment",
        extra_data: dict | None = None,
    ) -> InteractionRecord:
        """Send a single chat message and return a structured record."""
        payload = {"message": message, "session_id": session_id}
        if extra_data:
            payload.update(extra_data)

        url = f"{app.base_url}{app.chat_endpoint}"
        start = time.time()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.post(url, json=payload)
                    latency_ms = (time.time() - start) * 1000
                    response_data = resp.json() if resp.status_code == 200 else {}
                    return InteractionRecord(
                        app_name=app.name,
                        payload=message,
                        response=response_data,
                        status_code=resp.status_code,
                        latency_ms=latency_ms,
                    )
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    if attempt < self.max_retries:
                        logger.warning(f"Attempt {attempt+1} failed for {app.name}: {e}. Retrying...")
                        await asyncio.sleep(2.0 * (attempt + 1))
                    else:
                        latency_ms = (time.time() - start) * 1000
                        return InteractionRecord(
                            app_name=app.name,
                            payload=message,
                            response={},
                            status_code=-1,
                            latency_ms=latency_ms,
                            error=str(e),
                        )

    async def run_batch(
        self,
        app: TargetApp,
        payloads: list[str],
        session_id: str = "experiment",
        delay_between_ms: float = 500.0,
    ) -> list[InteractionRecord]:
        """Send a batch of payloads sequentially (preserves session state for App C)."""
        records = []
        for i, payload in enumerate(payloads):
            record = await self.send_message(app, payload, session_id=session_id)
            records.append(record)
            logger.debug(f"[{app.name}] [{i+1}/{len(payloads)}] sent: '{payload[:60]}'")
            if delay_between_ms > 0:
                await asyncio.sleep(delay_between_ms / 1000.0)
        return records

    async def reset_app(self, app: TargetApp, session_id: str = "experiment") -> bool:
        """Reset application state between experiment runs."""
        url = f"{app.base_url}{app.reset_endpoint}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.post(url, json={"session_id": session_id})
                return resp.status_code == 200
            except Exception as e:
                logger.warning(f"Reset failed for {app.name}: {e}")
                return False

    async def check_health(self, app: TargetApp) -> bool:
        """Verify that a benchmark app is healthy before running experiments."""
        url = f"{app.base_url}{app.health_endpoint}"
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.get(url)
                return resp.status_code == 200
            except Exception:
                return False

    async def check_all_healthy(self) -> dict[str, bool]:
        """Check health of all registered benchmark apps."""
        results = {}
        for key, app in BENCHMARK_APPS.items():
            healthy = await self.check_health(app)
            results[key] = healthy
            status = "✅ HEALTHY" if healthy else "❌ UNREACHABLE"
            logger.info(f"  {app.name}: {status}")
        return results
