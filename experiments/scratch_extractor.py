import json
import re

campaign_file = "c:/Users/lucag/PenTesLLM/experiments/results/mas/campaign_CHAIN_2_ADAPTIVE_1782652119.json"
traces_file = "c:/Users/lucag/PenTesLLM/experiments/results/mas/traces.jsonl"
out_file = "c:/Users/lucag/PenTesLLM/experiments/scratch_extraction.json"

with open(campaign_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
    successful_trial_ids = set(r['trial_id'] for r in data['results'] if r.get('success'))

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

results = []
for trace in last_30:
    trial_id = trace.get('trial_id')
    if trial_id in successful_trial_ids:
        try:
            # Analyze L2 to find the turn
            l2_agent = trace.get('agents', {}).get('Senior System Administrator', {})
            responses = l2_agent.get('raw_llm_responses', [])
            
            turn_idx = -1
            for i, r in enumerate(responses):
                r_str = json.dumps(r)
                if "tool_calls" in r_str or "function_call" in r_str or "write_file" in r_str:
                    turn_idx = i
                    break
            
            turn_num = turn_idx + 1 if turn_idx != -1 else -1
            
            events = trace.get('events', [])
            # L1 to L2 messages
            l1_responses = []
            l2_responses = [] # L2 to L1 messages (objections)
            for e in events:
                if e.get('event_type') == 'LLM_RESPONSE' and e.get('agent_role') == 'Senior Security Researcher':
                    raw_out = e.get('data', {}).get('raw_output', '')
                    l1_responses.append(raw_out)
                elif e.get('event_type') == 'LLM_RESPONSE' and e.get('agent_role') == 'Senior System Administrator':
                    raw_out = e.get('data', {}).get('raw_output', '')
                    l2_responses.append(raw_out)
            
            l1_msg = l1_responses[turn_idx] if turn_idx != -1 and turn_idx < len(l1_responses) else "N/A"
            l2_obj = l2_responses[turn_idx - 1] if turn_idx > 0 and turn_idx - 1 < len(l2_responses) else "N/A"
            
            results.append({
                'trial_id': trial_id,
                'turn': turn_num,
                'l2_objection': str(l2_obj),
                'l1_payload': str(l1_msg)
            })
        except Exception as e:
            print("Error processing", trial_id, e)

with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"Extracted {len(results)} items to {out_file}")
