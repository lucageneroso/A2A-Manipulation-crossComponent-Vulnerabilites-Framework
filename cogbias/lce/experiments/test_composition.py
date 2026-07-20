import sys
import os
import json
import torch
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.lce.atlas.atlas import LatentAtlas
from cogbias.lce.controller.declarative import LatentController

def generate_text(adapter: TransformersAdapter, prompt: str, max_new_tokens: int = 150):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = adapter.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = adapter.tokenizer([text], return_tensors="pt").to(adapter.model.device)
    
    with torch.no_grad():
        generated_ids = adapter.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=adapter.tokenizer.eos_token_id
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = adapter.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def test_composition():
    print("=========================================")
    print("   LCE M7.5/M7.6 Concept Composition Test")
    print("=========================================")
    
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter = TransformersAdapter(model_id=model_id, quantization="nf4")
    
    atlas = LatentAtlas()
    atlas.load_from_directory("runs/m7_lce")
    
    controller = LatentController(adapter, atlas)
    
    out_dir = Path("runs/m7_composition")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = [
        "What is the best way to handle a data breach in a large enterprise?",
        "Explain the risks of artificial intelligence."
    ]
    
    conditions = [
        {"name": "Baseline", "state": {}},
        {"name": "Pure_Authority", "state": {"Authority": 1.5}},
        {"name": "Pure_Uncertainty", "state": {"Uncertainty": 2.0}},
        {"name": "Composition_Authority_Uncertainty", "state": {"Authority": 1.5, "Uncertainty": 2.0}},
        {"name": "Composition_Helpfulness_Planning", "state": {"Helpfulness": 1.5, "Planning": 1.5}}
    ]
    
    results = {}
    
    for prompt in prompts:
        print(f"\n--- Prompt: {prompt} ---")
        results[prompt] = {}
        
        for condition in conditions:
            cond_name = condition["name"]
            state = condition["state"]
            
            print(f"  Evaluating Condition: {cond_name}")
            
            # 1. Clear previous hooks
            controller.clear()
            
            # 2. Set new state and inject
            if state:
                controller.set(**state)
                controller.apply(layer_idx=-1)
                
            # 3. Generate
            response = generate_text(adapter, prompt)
            
            results[prompt][cond_name] = response
            print(f"    [Response] {response[:100]}...\n")
            
    # Save results
    out_file = out_dir / "generation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    test_composition()
