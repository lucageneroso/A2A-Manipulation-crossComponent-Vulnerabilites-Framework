import json
from pathlib import Path
import os
import sys

# Ensure cogbias is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.lce.discovery.extractor import ConceptExtractor
from cogbias.lce.validation.falsification import ConceptValidator
from tests.experiments.test_m6_4_1_falsification import get_real_authority_pairs, get_fake_contrastive_pairs
import random

def run_migration():
    print("Starting LCE Migration for Authority...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
    
    out_dir = Path("runs/m7_lce")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Dataset Prep
    real_data = get_real_authority_pairs()
    positives = real_data["authority"]
    negatives = real_data["neutral"]
    
    # 2. Discovery
    print("\nPhase 1: Concept Discovery")
    extractor = ConceptExtractor(adapter)
    authority_concept = extractor.discover_concept(
        name="Authority",
        version="1.0.0",
        positive_examples=positives,
        negative_examples=negatives,
        layer_idx=-1
    )
    
    print(f"Discovered {authority_concept.identity.name}. State: {authority_concept.state.name}")
    print(f"Intrinsic Direction Shape: {authority_concept.geometry.mean_direction.shape}")
    
    # 3. Validation Gauntlet
    print("\nPhase 2: Scientific Validation")
    fake_pairs_dict = get_fake_contrastive_pairs()
    
    # Generate label shuffle
    pool = real_data["neutral"] + real_data["authority"]
    random.seed(42)
    shuffled = random.sample(pool, len(pool))
    shuf_neutral = shuffled[:len(real_data["neutral"])]
    shuf_auth = shuffled[len(real_data["neutral"]):]
    fake_pairs_dict["Label_Shuffle"] = list(zip(shuf_neutral, shuf_auth))
    
    validator = ConceptValidator(adapter)
    authority_concept, report = validator.validate(
        concept=authority_concept,
        fake_pairs=fake_pairs_dict,
        layer_idx=-1,
        n_bootstraps=50,
        cosine_threshold=0.6
    )
    
    print(f"\nValidation Report: {json.dumps(report, indent=2)}")
    
    print(f"\nFinal Concept State: {authority_concept.state.name}")
    
    # 4. Serialization
    out_file = out_dir / f"{authority_concept.identity.name}_v{authority_concept.identity.version}.lce"
    authority_concept.save(str(out_file))
    print(f"Serialized concept to {out_file}")
    
if __name__ == "__main__":
    run_migration()
