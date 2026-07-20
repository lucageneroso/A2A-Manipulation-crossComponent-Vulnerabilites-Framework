import json

traces_file = "c:/Users/lucag/PenTesLLM/experiments/results/mas/traces.jsonl"
adaptive_chain2_traces = []
with open(traces_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        try:
            trace = json.loads(line)
            config = trace.get('config_name', '')
            if 'CHAIN_2' in config and 'adaptive' in config.lower():
                adaptive_chain2_traces.append(trace)
        except Exception as e:
            pass

last_30 = adaptive_chain2_traces[-30:]

for trace in last_30:
    if trace.get('trial_id') == 2:
        events = trace.get('events', [])
        for i, e in enumerate(events[:10]):
            print(f"Event {i}: type={e.get('event_type')}, role={e.get('agent_role')}")
            if e.get('data'):
                print(f"   data keys: {list(e.get('data').keys())}")
        break
