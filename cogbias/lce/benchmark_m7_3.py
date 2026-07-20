import json
from pathlib import Path
import os
import sys
import random

# Ensure cogbias is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.lce.discovery.extractor import ConceptExtractor
from cogbias.lce.validation.falsification import ConceptValidator
from tests.experiments.test_m6_4_1_falsification import get_fake_contrastive_pairs
from cogbias.lce.benchmarks.benchmark import get_standardized_benchmarks

def run_benchmark():
    print("Starting LCE Benchmark M7.3...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
    
    out_dir = Path("runs/m7_lce")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = ConceptExtractor(adapter)
    validator = ConceptValidator(adapter)
    
    benchmark_data = get_standardized_benchmarks()
    # Filter to only test Uncertainty since others passed
    benchmark_data = {"Uncertainty": benchmark_data["Uncertainty"]}
    base_fake_pairs = get_fake_contrastive_pairs()
    
    for concept_name, data in benchmark_data.items():
        print(f"\n=====================================")
        print(f"Processing Concept: {concept_name}")
        print(f"=====================================")
        
        positives = data["positives"]
        negatives = data["negatives"]
        
        # 1. Discovery
        concept = extractor.discover_concept(
            name=concept_name,
            version="1.0.0",
            positive_examples=positives,
            negative_examples=negatives,
            layer_idx=-1
        )
        print(f"Discovered {concept.identity.name}.")
        
        # 2. Setup Controls
        fake_pairs_dict = dict(base_fake_pairs) # copy
        pool = negatives + positives
        random.seed(42)
        shuffled = random.sample(pool, len(pool))
        shuf_neg = shuffled[:len(negatives)]
        shuf_pos = shuffled[len(negatives):]
        fake_pairs_dict["Label_Shuffle"] = list(zip(shuf_neg, shuf_pos))
        
        # 3. Validation
        concept, report = validator.validate(
            concept=concept,
            fake_pairs=fake_pairs_dict,
            layer_idx=-1,
            n_bootstraps=50,
            cosine_threshold=0.6
        )
        
        print(f"\nValidation Report: {json.dumps(report, indent=2)}")
        
        # 4. Serialization
        out_file = out_dir / f"{concept.identity.name}_v{concept.identity.version}.lce"
        concept.save(str(out_file))
        print(f"Saved {concept_name} to {out_file}")

if __name__ == "__main__":
    run_benchmark()
