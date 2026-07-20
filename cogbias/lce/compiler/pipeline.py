import json
from typing import Dict, Any, List

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.lce.atlas.atlas import LatentAtlas
from cogbias.lce.controller.declarative import LatentController

class SemanticCompiler:
    """
    Translates human intent (e.g., "senior cybersecurity architect") into normalized LCE coordinates.
    Acts as the bridging layer between prompt engineering and latent engineering.
    """
    def __init__(self, adapter: TransformersAdapter, atlas: LatentAtlas, controller: LatentController):
        self.adapter = adapter
        self.atlas = atlas
        self.controller = controller
        
        self.last_compilation_state = {}
        self.last_explanations = {}

    def _mock_llm_parse(self, intent: str) -> Dict[str, Any]:
        """
        In a full implementation, this uses a structured generation prompt on the LLM
        to map the intent into the available Atlas concepts.
        For this prototype, we use a simple keyword heuristic mapping to demonstrate architecture.
        """
        intent_lower = intent.lower()
        coords = {}
        explanations = {}
        
        # Heuristic rules for prototyping
        if "senior" in intent_lower or "architect" in intent_lower or "lead" in intent_lower:
            coords["Authority"] = 0.8
            explanations["Authority"] = "High because: leadership requirement, decision ownership, responsibility."
            
        if "architect" in intent_lower or "engineer" in intent_lower or "planner" in intent_lower:
            coords["Planning"] = 0.9
            explanations["Planning"] = "High because: architecture, long-term design, structural thinking."
            
        if "assistant" in intent_lower or "support" in intent_lower or "help" in intent_lower:
            coords["Helpfulness"] = 0.9
            explanations["Helpfulness"] = "High because: primary role is assistive and cooperative."
        else:
            # Baseline helpfulness for non-assistive roles
            coords["Helpfulness"] = 0.4
            explanations["Helpfulness"] = "Moderate because: standard cooperative baseline."
            
        if "research" in intent_lower or "analyst" in intent_lower:
            coords["Uncertainty"] = 0.6
            explanations["Uncertainty"] = "Moderate-High because: exploratory role requiring epistemic humility."
        else:
            coords["Uncertainty"] = 0.2
            explanations["Uncertainty"] = "Low because: role expects decisive, factual execution."
            
        return {"coordinates": coords, "explanations": explanations}

    def compile(self, intent: str, auto_apply: bool = False) -> Dict[str, float]:
        """
        Compiles the human intent into latent coordinates.
        """
        print(f"[SemanticCompiler] Compiling intent: '{intent}'")
        parsed = self._mock_llm_parse(intent)
        
        self.last_compilation_state = parsed["coordinates"]
        self.last_explanations = parsed["explanations"]
        
        if auto_apply:
            self.controller.clear()
            self.controller.set(**self.last_compilation_state)
            self.controller.apply(layer_idx=-1)
            print(f"[SemanticCompiler] Injected {self.last_compilation_state} into LatentController.")
            
        return self.last_compilation_state

    def explain(self) -> str:
        """
        Returns a human-readable explanation of the compilation process.
        """
        if not self.last_explanations:
            return "No compilation has been run yet."
            
        explanation_blocks = []
        for concept, weight in self.last_compilation_state.items():
            reason = self.last_explanations.get(concept, "No explanation provided.")
            explanation_blocks.append(f"\"{concept}\" weight ({weight}):\n- {reason}")
            
        return "\n\n".join(explanation_blocks)
