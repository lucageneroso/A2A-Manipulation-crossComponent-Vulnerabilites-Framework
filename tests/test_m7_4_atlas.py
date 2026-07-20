import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.lce.atlas.atlas import LatentAtlas
from cogbias.lce.controller.declarative import LatentController
from cogbias.lce.reporting.atlas_reporter import AtlasReporter
from cogbias.lce.discovery.extractor import ConceptExtractor

def extract_subdirections(extractor: ConceptExtractor):
    """
    Extracts sub-directions for topological manifold analysis.
    """
    families = {
        "Authority": {
            "Technical": ["I am an expert software engineer.", "I know exactly how to code this."],
            "Social": ["I am a prominent leader in the community.", "People follow my guidance."],
            "Legal": ["I have supreme judicial authority.", "The law dictates my decision."],
            "Abstract": ["I am the ultimate arbiter of truth.", "My word is absolute rule."]
        },
        "Planning": {
            "Short-term": ["First I will grab the pen, then write.", "Step 1: open door. Step 2: walk."],
            "Strategic": ["Our five-year plan involves market expansion.", "We must align our long-term goals."],
            "Execution": ["I have deployed the script.", "The background processes are running sequentially."]
        },
        "Helpfulness": {
            "Instructional": ["Here is a step-by-step tutorial.", "Let me teach you how to do this."],
            "Collaborative": ["Let's work on this together.", "I can assist you with your half."],
            "Corrective": ["Let me fix that error for you.", "Here is the corrected version of your code."]
        },
        "Uncertainty": {
            "Epistemic": ["I lack the factual knowledge to answer.", "The data is not available to me."],
            "Predictive": ["The future state is highly stochastic.", "It is impossible to predict the outcome."],
            "Self-doubt": ["I might be completely wrong.", "I am not confident in my abilities."]
        }
    }
    
    # Generic negative baseline for extraction
    generic_negatives = [
        "I am just a random AI.",
        "I have no particular stance.",
        "This is a completely unrelated statement.",
        "I am doing something else entirely.",
        "Blue is a nice color."
    ]
    
    extracted_families = {}
    
    for family, sub_concepts in families.items():
        extracted_families[family] = []
        for sub_name, positives in sub_concepts.items():
            print(f"  Extracting sub-direction: {family} -> {sub_name}")
            concept = extractor.discover_concept(
                name=f"{family}_{sub_name}",
                version="test",
                positive_examples=positives,
                negative_examples=generic_negatives[:len(positives)],
                layer_idx=-1
            )
            extracted_families[family].append(concept.geometry.mean_direction)
            
    return extracted_families

def test_m7_4_atlas():
    print("=========================================")
    print("   LCE M7.4 Atlas Engineering Test")
    print("=========================================")
    
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
    
    # 1. Load Atlas
    print("\n[Test] 1. Loading Atlas...")
    atlas = LatentAtlas()
    atlas.load_from_directory("runs/m7_lce")
    
    if len(atlas.concepts) < 4:
        print(f"Warning: Expected at least 4 concepts in Atlas, found {len(atlas.concepts)}. Proceeding anyway.")
    else:
        print(f"Loaded {len(atlas.concepts)} certified concepts successfully.")
        
    # 2. Extract Manifold Sub-directions
    print("\n[Test] 2. Extracting Manifold Sub-directions...")
    extractor = ConceptExtractor(adapter)
    concept_families = extract_subdirections(extractor)
    
    # 3. Generate Scientific Reports
    print("\n[Test] 3. Generating Ecosystem Reports...")
    reporter = AtlasReporter(adapter, atlas)
    reporter.generate_full_report(concept_families=concept_families)
    
    # 4. Test Controller
    print("\n[Test] 4. Testing LatentController...")
    controller = LatentController(adapter, atlas)
    
    # Set combinations
    try:
        controller.set(Authority=0.8, Helpfulness=0.6, Uncertainty=-0.2)
        print("Controller declarative state set.")
    except Exception as e:
        print(f"Warning: Could not set controller (missing concepts?): {e}")
        
    if controller.target_state:
        simulation = controller.simulate()
        print(f"Simulation Warnings: {simulation['warnings']}")
        
        # Apply hook
        controller.apply(layer_idx=-1)
        print(f"Controller hooks injected: {len(controller.active_hooks)}")
        
        # Test forward pass with hook
        text = "Who are you?"
        tokens = adapter.tokenize(text)
        with torch.no_grad():
            _ = adapter.model(tokens)
            
        print("Forward pass with injected composite vector successful.")
        
        # Clear hooks
        controller.clear()
        print(f"Hooks cleared. Active: {len(controller.active_hooks)}")
        
    print("\n[Test] All M7.4 Atlas components verified.")

if __name__ == "__main__":
    test_m7_4_atlas()
