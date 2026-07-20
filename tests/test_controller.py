import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.lce.atlas.atlas import LatentAtlas
from cogbias.lce.controller.declarative import LatentController

def test_controller():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
    
    atlas = LatentAtlas()
    atlas.load_from_directory("runs/m7_lce")
    
    controller = LatentController(adapter, atlas)
    controller.set(Authority=0.8, Helpfulness=0.6, Uncertainty=-0.2)
    controller.apply(layer_idx=-1)
    
    text = "Who are you?"
    tokens = adapter.tokenize(text)
    with torch.no_grad():
        _ = adapter.model(**tokens)
        
    print("LatentController applied successfully!")

if __name__ == "__main__":
    test_controller()
