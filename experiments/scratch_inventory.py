import json
import glob
from pathlib import Path
from collections import Counter
import os

mas_dir = "c:/Users/lucag/PenTesLLM/experiments/results/mas"
files = glob.glob(f"{mas_dir}/campaign_CHAIN_*_ADAPTIVE_*.json")

inventory = []

for f in sorted(files):
    name = Path(f).name
    try:
        data = json.load(open(f, 'r', encoding='utf-8'))
        results = data.get('results', [])
        total = data.get('total_trials', len(results))
        successes = data.get('successes', 0) # EAPE successes, often 0 if EAPE path not full. Or sometimes 'yields'. We should count `tool_call_executed`
        
        yields = 0
        ards = 0
        biases = Counter()
        
        for r in results:
            if r.get('tool_call_executed'):
                yields += 1
            if r.get('has_disconnect'):
                ards += 1
            # Infer bias from payload name or something else
            bias_str = "unknown"
            if "payload_name" in r:
                bias_str = r["payload_name"]
            biases[bias_str] += 1
            
        inventory.append({
            'file': name,
            'ts': data.get('timestamp', os.path.getmtime(f)),
            'total': total,
            'yields': yields,
            'ards': ards,
            'biases': dict(biases),
        })
    except Exception as e:
        print(f"Error reading {name}: {e}")

print("=== JSON FILES INVENTORY ===")
for item in inventory:
    print(f"File: {item['file']}")
    print(f"  Trials: {item['total']} | Yields: {item['yields']} | ARD: {item['ards']}")
    print(f"  Biases: {item['biases']}")
    print("-" * 40)

# Check traces.jsonl
traces_path = Path(f"{mas_dir}/traces.jsonl")
print("\n=== TRACES.JSONL ANALYSIS ===")
adaptive_trials_traces = []
try:
    for line in open(traces_path, "r", encoding="utf-8"):
        if not line.strip(): continue
        try:
            data = json.loads(line)
            config = data.get("config_name", "")
            if "CHAIN_" in config and "adaptive" in config.lower():
                adaptive_trials_traces.append(data)
        except:
            pass
    print(f"Total adaptive trials in traces.jsonl: {len(adaptive_trials_traces)}")
    
    # Try to group by timestamp or run
    runs = Counter()
    for t in adaptive_trials_traces:
        # trace usually has timestamp
        ts = t.get('metadata', {}).get('timestamp', 'unknown')
        if ts == 'unknown': ts = t.get('timestamp', 'unknown')
        runs[ts] += 1
    
    # Check if traces is an aggregator
    print(f"Number of unique timestamps (runs) in traces: {len(runs)}")
    
except Exception as e:
    print(f"Error parsing traces.jsonl: {e}")

