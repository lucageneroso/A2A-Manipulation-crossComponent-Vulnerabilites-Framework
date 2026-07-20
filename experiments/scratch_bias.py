import json
from pathlib import Path

traces_path = Path("c:/Users/lucag/PenTesLLM/experiments/results/mas/traces.jsonl")

adaptive_trials = []
for line in open(traces_path, "r", encoding="utf-8"):
    if not line.strip(): continue
    try:
        data = json.loads(line)
        config = data.get("config_name", "")
        if "CHAIN_" in config and "adaptive" in config.lower():
            adaptive_trials.append(data)
    except:
        pass

# 30-trial detailed campaign file:
file_30 = "c:/Users/lucag/PenTesLLM/experiments/results/mas/campaign_CHAIN_2_ADAPTIVE_1782652119.json"
data_30 = json.load(open(file_30, 'r', encoding='utf-8'))
timestamps_30 = set(r['timestamp'] for r in data_30['results'])

def analyze_bias(trials):
    biases = {'Authority': 0, 'Urgency': 0, 'Social Proof': 0}
    for trial in trials:
        l2_trace = trial.get("agents", {}).get("Senior System Administrator", {})
        responses = l2_trace.get("raw_llm_responses", [])
        yield_turn = -1
        for i, r in enumerate(responses):
            if "ChatCompletionMessageFunctionToolCall" in r or "tool_calls" in r:
                yield_turn = i + 1
                break
        if yield_turn != -1:
            bias_map = {1: "Authority", 2: "Urgency", 3: "Social Proof"}
            b = bias_map.get(min(yield_turn, 3), "N/A")
            if b in biases: biases[b] += 1
    return biases

# What if we only take the 29 trials from 10/19 + 3/10?
# The 30 trials from 1782652119 are:
trials_30 = [t for t in adaptive_trials if t['metadata']['timestamp'] in timestamps_30]
print("30 Trials Bias:", analyze_bias(trials_30))
print("All Adaptive Bias:", analyze_bias(adaptive_trials))
