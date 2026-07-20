import torch
import numpy as np
from typing import Dict, Any

from cogbias.lce.core.concept import LatentConcept, ConceptIdentity, ConceptSemantics
from cogbias.lce.compiler.lcir import LatentConceptIR

class InvalidLCIRError(Exception):
    pass

class BackendCompiler:
    """
    Compiles an LCIR into a target-specific native Latent Concept (.lce).
    Includes Adversarial LCIR Validation to reject unsafe or impossible concepts.
    """
    def __init__(self):
        pass

    def _validate_lcir(self, lcir: LatentConceptIR):
        """Adversarial LCIR Validation."""
        print(f"[BackendCompiler] Validating LCIR for {lcir.concept_name}...")
        
        # 1. Semantic Contradiction Check
        dims = lcir.semantic.dimensions
        if "decision_confidence" in dims and "uncertainty_tolerance" in dims:
            if dims["decision_confidence"] > 0.8 and dims["uncertainty_tolerance"] > 0.8:
                raise InvalidLCIRError("Semantic Contradiction: High confidence and high uncertainty tolerance.")
                
        # 2. Causal Contradiction Check
        inc = set(lcir.causal.must_increase)
        not_inc = set(lcir.causal.must_not_increase)
        overlap = inc.intersection(not_inc)
        if overlap:
            raise InvalidLCIRError(f"Causal Contradiction: Overlap in constraints: {overlap}")
            
        # 3. Impossible Geometry Bounds Check
        bounds = lcir.geometric.intrinsic_dimension_bounds
        if len(bounds) != 2 or bounds[0] > bounds[1] or bounds[0] < 1:
            raise InvalidLCIRError(f"Impossible Geometry: Invalid intrinsic dimension bounds: {bounds}")
            
        print("[BackendCompiler] LCIR Validation Passed.")

    def compile(self, lcir: LatentConceptIR, target_model_name: str, target_layer: int, target_dim: int = 2048) -> LatentConcept:
        """
        Compiles the validated LCIR into a native tensor for the target model.
        In a real scenario, this uses the target model's inverse mapping or a pre-trained regression.
        """
        self._validate_lcir(lcir)
        
        print(f"[BackendCompiler] Compiling {lcir.concept_name} for {target_model_name} (Layer {target_layer})...")
        
        # Mock compilation projection based on semantic dimensions
        # Real implementation would multiply a behavioral regression matrix by the semantic dimensions
        np.random.seed(hash(target_model_name + lcir.concept_name) % 10000)
        base_vector = np.random.randn(target_dim)
        
        # Scale by dominant semantic dimension
        scale = max(lcir.semantic.dimensions.values()) if lcir.semantic.dimensions else 1.0
        compiled_vector = base_vector * scale
        
        direction_tensor = torch.tensor(compiled_vector, dtype=torch.float32)
        
        identity = ConceptIdentity(
            name=lcir.concept_name,
            version="1.0.0",
            model_hash=target_model_name,
            extraction_protocol="LCIR_Compiled"
        )
        
        concept = LatentConcept(identity=identity, semantics=ConceptSemantics())
        concept.geometry.mean_direction = compiled_vector
        concept.causality.intervention_layers = [target_layer]
        concept.geometry.intrinsic_dimension = lcir.geometric.intrinsic_dimension_bounds[0]
        
        print(f"[BackendCompiler] Compilation successful: {lcir.concept_name}_Compiled_{target_model_name}.lce")
        return concept
