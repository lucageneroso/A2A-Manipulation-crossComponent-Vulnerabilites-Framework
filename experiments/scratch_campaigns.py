import json
import glob
from pathlib import Path

files = glob.glob('c:/Users/lucag/PenTesLLM/experiments/results/mas/campaign_*.json')
for f in files:
    try:
        data = json.load(open(f, 'r', encoding='utf-8'))
        print(f"{Path(f).name}: topology={data.get('topology')}, trials={data.get('total_trials')}, successes={data.get('successes')}")
    except Exception as e:
        print(f"Error {f}: {e}")
