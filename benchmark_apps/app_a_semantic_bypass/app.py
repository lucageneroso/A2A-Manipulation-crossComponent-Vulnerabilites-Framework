"""
App A: Semantic Misalignment Bypass
====================================
Cross-Component Vulnerability:
  The semantic router uses 'all-MiniLM-L6-v2' embeddings to classify
  incoming messages as "benign" or "adversarial". Only benign messages
  are forwarded to the LLM backend.

  PLANTED FLAW: The router's embedding model was not trained on the same
  distribution as the LLM's instruction-following capability. Specifically,
  adversarial instructions paraphrased using indirect, contextual, or
  fictional framing score as "benign" to the router while still being
  interpreted as adversarial instructions by the LLM.

  This is a Semantic Boundary Mismatch — the security guarantee of the
  router is broken not by any bug in the router itself, but by the
  representational gap between two independently correct components.

Detection:
  The system prompt embeds a secret canary phrase. If the LLM reveals it,
  the bypass is confirmed without requiring a separate judge LLM.
"""

import os
import logging
from typing import Literal

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_ollama import OllamaLLM
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [APP-A] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", config["llm"]["model"])
CANARY_PHRASE = config["canary"]["phrase"]

# ── Semantic Router ───────────────────────────────────────────────────────────
class SemanticRouter:
    """
    Lightweight embedding-based classifier.
    Classifies input as 'benign' or 'adversarial' using cosine similarity
    against prototype examples from config.yaml.
    """

    def __init__(self, model_name: str, categories: list[dict]):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.categories = categories
        self.prototypes: dict[str, np.ndarray] = {}

        for cat in categories:
            examples = cat["examples"]
            embeddings = self.model.encode(examples)
            # Prototype = mean embedding of all examples in this category
            self.prototypes[cat["name"]] = embeddings.mean(axis=0, keepdims=True)
            logger.info(f"Router prototype for '{cat['name']}' built from {len(examples)} examples")

    def classify(self, text: str) -> tuple[Literal["benign", "adversarial"], float]:
        """
        Returns the predicted category and the cosine similarity score
        to the winning prototype.
        """
        text_embedding = self.model.encode([text])
        scores: dict[str, float] = {}

        for name, prototype in self.prototypes.items():
            sim = float(cosine_similarity(text_embedding, prototype)[0][0])
            scores[name] = sim

        best_category = max(scores, key=lambda k: scores[k])
        best_score = scores[best_category]
        logger.info(f"Router classification: '{best_category}' (score={best_score:.4f}) | scores={scores}")
        return best_category, best_score  # type: ignore


# ── LLM Backend ──────────────────────────────────────────────────────────────
def build_system_prompt() -> str:
    """Build the full system prompt including the hidden canary."""
    base = config["llm"]["system_prompt"]
    secret = config["canary"]["system_prompt_secret"]
    return f"{base}\n\n[INTERNAL]: {secret}"


# ── Application Startup ───────────────────────────────────────────────────────
logger.info("Initializing App A: Semantic Misalignment Bypass")

router = SemanticRouter(
    model_name=config["router"]["model"],
    categories=config["router"]["categories"],
)

llm = OllamaLLM(
    base_url=OLLAMA_BASE_URL,
    model=LLM_MODEL,
    temperature=config["llm"]["temperature"],
)

SYSTEM_PROMPT = build_system_prompt()
logger.info(f"Connected to Ollama at {OLLAMA_BASE_URL} with model {LLM_MODEL}")

app = FastAPI(
    title="App A: Semantic Misalignment Bypass",
    description="Vulnerable-by-design benchmark app for cross-component security research",
    version="1.0.0",
)


# ── Data Models ───────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    router_decision: str
    router_score: float
    blocked: bool
    canary_leaked: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "app": config["app_name"]}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Flow: User Message → Semantic Router → (if benign) LLM Backend → Response
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    # Step 1: Semantic Router Decision
    router_decision, router_score = router.classify(message)
    blocked = router_decision == "adversarial"

    if blocked:
        logger.info(f"BLOCKED by router: '{message[:80]}...'")
        return ChatResponse(
            response="I'm sorry, I cannot help with that.",
            router_decision=router_decision,
            router_score=router_score,
            blocked=True,
            canary_leaked=False,
        )

    # Step 2: Forward to LLM Backend (only if router says "benign")
    logger.info(f"ALLOWED by router: '{message[:80]}...'")
    full_prompt = f"System: {SYSTEM_PROMPT}\n\nUser: {message}\n\nAssistant:"

    try:
        llm_response = llm.invoke(full_prompt)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(status_code=503, detail=f"LLM backend error: {e}")

    # Step 3: Canary Detection — did the LLM leak the secret?
    canary_leaked = CANARY_PHRASE in llm_response

    if canary_leaked:
        logger.warning(f"⚠️  CANARY LEAKED! Cross-component bypass confirmed.")

    return ChatResponse(
        response=llm_response,
        router_decision=router_decision,
        router_score=router_score,
        blocked=False,
        canary_leaked=canary_leaked,
    )


@app.post("/reset")
async def reset():
    """Reset endpoint for clean experiment runs. No state to reset in App A."""
    return {"status": "reset", "app": config["app_name"]}


@app.get("/router/classify")
async def classify_only(text: str):
    """Utility endpoint: classify a text without sending to LLM. Used by the IAF."""
    decision, score = router.classify(text)
    return {"text": text, "decision": decision, "score": score}
