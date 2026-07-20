import torch
import numpy as np
from typing import List, Tuple

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.lce.core.concept import LatentConcept, ConceptIdentity, ConceptSemantics
from cogbias.lce.core.state import ConceptState

class ConceptExtractor:
    """
    Concept-agnostic extraction pipeline.
    Discovers the geometric direction of a latent concept.
    """
    def __init__(self, adapter: TransformersAdapter):
        self.adapter = adapter
        
    def discover_concept(
        self, 
        name: str, 
        version: str,
        positive_examples: List[str], 
        negative_examples: List[str],
        layer_idx: int = -1,
        extraction_protocol: str = "mean_difference"
    ) -> LatentConcept:
        """
        Extracts the latent concept representation.
        Returns a LatentConcept in the DISCOVERED state.
        """
        if len(positive_examples) != len(negative_examples):
            raise ValueError("Must provide equal number of positive and negative examples for contrastive extraction.")
            
        pairs = list(zip(negative_examples, positive_examples))
        
        # Identity
        identity = ConceptIdentity(
            name=name,
            version=version,
            model_hash=self.adapter.model_id,  # using ID as hash for now
            extraction_protocol=extraction_protocol
        )
        
        # Semantics
        semantics = ConceptSemantics(
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            domain_coverage=["general"] # Can be expanded later
        )
        
        concept = LatentConcept(identity, semantics)
        
        # Extraction
        if extraction_protocol == "mean_difference":
            mean_dir, cov = self._extract_mean_difference(pairs, layer_idx)
            concept.geometry.mean_direction = mean_dir
            concept.geometry.covariance = cov
        else:
            raise NotImplementedError(f"Protocol {extraction_protocol} not implemented.")
            
        concept.transition_state(ConceptState.DISCOVERED)
        
        return concept
        
    def _extract_mean_difference(self, pairs: List[Tuple[str, str]], layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        diffs = []
        for base, target in pairs:
            # Base (Negative)
            inp_b = self.adapter.tokenize(base)
            diag_b = self.adapter.forward_diagnostic(inp_b)
            h_b = diag_b["hidden_states"][layer_idx]
            v_b = h_b[0, -1, :].clone().detach().to(torch.float32).cpu().numpy()
            
            # Target (Positive)
            inp_t = self.adapter.tokenize(target)
            diag_t = self.adapter.forward_diagnostic(inp_t)
            h_t = diag_t["hidden_states"][layer_idx]
            v_t = h_t[0, -1, :].clone().detach().to(torch.float32).cpu().numpy()
            
            diffs.append(v_t - v_b)
            
        diffs = np.array(diffs)
        mean_dir = np.mean(diffs, axis=0)
        
        # Calculate covariance of the difference vectors to capture geometric spread
        cov = np.cov(diffs, rowvar=False) if len(diffs) > 1 else np.zeros((len(mean_dir), len(mean_dir)))
        
        return mean_dir, cov
