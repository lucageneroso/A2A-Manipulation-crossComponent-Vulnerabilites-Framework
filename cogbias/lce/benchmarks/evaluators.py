import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseEvaluator(ABC):
    """
    Interface for evaluating model outputs in the LCE Benchmark Suite.
    """
    @abstractmethod
    def evaluate(self, prompt: str, generated_text: str, expected_concept: str = None) -> Dict[str, Any]:
        """
        Evaluates the generated text and returns a dictionary of scores.
        """
        pass

class RuleBasedEvaluator(BaseEvaluator):
    """
    Evaluates based on simple lexical heuristics and rules.
    """
    def __init__(self, target_keywords: List[str] = None, negative_keywords: List[str] = None):
        self.target_keywords = target_keywords or []
        self.negative_keywords = negative_keywords or []
        
    def evaluate(self, prompt: str, generated_text: str, expected_concept: str = None) -> Dict[str, Any]:
        text_lower = generated_text.lower()
        
        target_hits = sum(1 for kw in self.target_keywords if kw.lower() in text_lower)
        negative_hits = sum(1 for kw in self.negative_keywords if kw.lower() in text_lower)
        
        score = 0.0
        if self.target_keywords:
            score = target_hits / len(self.target_keywords)
            
        return {
            "evaluator": "RuleBased",
            "score": score,
            "target_hits": target_hits,
            "negative_hits": negative_hits
        }

class HumanEvaluationPlaceholder(BaseEvaluator):
    """
    Placeholder that outputs null scores for offline human review.
    """
    def evaluate(self, prompt: str, generated_text: str, expected_concept: str = None) -> Dict[str, Any]:
        return {
            "evaluator": "HumanPlaceholder",
            "score": None,
            "needs_review": True
        }

class LLMJudgeEvaluator(BaseEvaluator):
    """
    Uses an LLM (e.g., via TransformersAdapter) to score the output.
    """
    def __init__(self, adapter):
        self.adapter = adapter
        
    def evaluate(self, prompt: str, generated_text: str, expected_concept: str = None) -> Dict[str, Any]:
        # For prototype purposes, this could formulate a prompt to ask the model to rate itself.
        # In this stub, we return a mock structured response to keep the benchmark fast.
        
        # Real implementation would call adapter.generate() with a judge prompt.
        # e.g., "Score the following text from 1 to 10 on how authoritative it is."
        return {
            "evaluator": "LLMJudge",
            "score": 0.85, # Mock score
            "reasoning": "The response is highly structured and authoritative."
        }
