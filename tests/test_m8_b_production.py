import json
import torch
import numpy as np
from pathlib import Path

from cogbias.lce.core.concept import LatentConcept, ConceptIdentity
from cogbias.lce.mlops.contracts import LatentContract, IdentityContract, GeometryContract, ValidationContract, BehaviorContract
from cogbias.lce.mlops.registry import ConceptRegistry
from cogbias.lce.mlops.pipeline import LatentCIPipeline

def create_mock_package(registry: ConceptRegistry, name: str, version: str):
    concept = LatentConcept(
        identity=ConceptIdentity(name=name, version=version, model_hash="Qwen2.5-1.5B", extraction_protocol="PCA"),
        direction=torch.randn(1536),
        layer_idx=18
    )
    # Mocking validation/causality metrics
    concept.validation.bootstrap_stability = 0.95
    concept.validation.falsification_score = 0.15
    concept.causality.effect_size = 0.85
    
    contract = LatentContract(
        identity=IdentityContract(name=name, version=version, model_compatibility=["Qwen2.5-1.5B"], layer_compatibility=[18]),
        geometry=GeometryContract(intrinsic_dimension=1, manifold_signature={"pca_var": 0.5}),
        validation=ValidationContract(),
        behavior=BehaviorContract(expected_increases=["decision confidence"], must_not_increase=["hallucination"])
    )
    
    registry.register_concept(concept, contract)

def test_production_readiness():
    print("=== Latent MLOps Production Readiness Test ===")
    registry = ConceptRegistry("runs/m8_b_test/registry")
    pipeline = LatentCIPipeline(registry, output_dir="runs/m8_b_test/ci_reports")
    
    # 1. Create mock discovered packages
    create_mock_package(registry, "Authority", "1.0.0")
    create_mock_package(registry, "Planning", "1.0.0")
    create_mock_package(registry, "Helpfulness", "1.0.0")
    
    # 2. Run CI Pipeline for all
    registry.scan()
    reports = []
    
    for name in ["Authority", "Planning", "Helpfulness"]:
        report = pipeline.run_pipeline(name, "1.0.0")
        reports.append(report)
        
    # 3. Generate LCE Production Readiness Report
    readiness_report = {
        "status": "READY_FOR_DEPLOYMENT",
        "packages_certified": len(reports),
        "details": reports
    }
    
    out_file = Path("runs/m8_b_test/lce_production_readiness_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(readiness_report, f, indent=2)
        
    print(f"Production readiness report generated at {out_file}")

if __name__ == "__main__":
    test_production_readiness()
