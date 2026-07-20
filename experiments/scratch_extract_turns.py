import json

campaign_file = "c:/Users/lucag/PenTesLLM/experiments/results/mas/campaign_CHAIN_2_ADAPTIVE_1782652119.json"
traces_file = "c:/Users/lucag/PenTesLLM/experiments/results/mas/traces.jsonl"

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
            
            # Find all L1 responses
            events = trace.get('events', [])
            l1_responses = []
            for e in events:
                if e.get('event_type') == 'LLM_RESPONSE' and e.get('agent_role') == 'Senior Security Researcher':
                    # Sometimes raw_output is a string, sometimes a list, etc.
                    raw_out = e.get('data', {}).get('raw_output', '')
                    l1_responses.append(raw_out)
            
            message_text = "N/A"
            if turn_idx != -1 and turn_idx < len(l1_responses):
                message_text = l1_responses[turn_idx]
            
            # Check if bias label can be inferred from the message text or if we just default to Turn X bias
            bias_label = f"Unlabeled (Turn {turn_num})"
            # Heuristics for the biases as defined in the thesis:
            msg_lower = str(message_text).lower()
            if turn_num == 1:
                bias_label = "Authority (Turn 1)"
            elif turn_num == 2:
                bias_label = "Urgency (Turn 2)"
            elif turn_num == 3:
                bias_label = "Social Proof (Turn 3)"
            elif turn_num > 3:
                # Based on the text, try to guess
                if "ciso" in msg_lower or "policy" in msg_lower or "direttive" in msg_lower:
                    bias_label = "Authority (Adaptive)"
                elif "immediata" in msg_lower or "urgente" in msg_lower or "secondo conta" in msg_lower:
                    bias_label = "Urgency (Adaptive)"
                elif "altri amministratori" in msg_lower or "hanno già" in msg_lower or "tutti i dipartimenti" in msg_lower:
                    bias_label = "Social Proof (Adaptive)"
                else:
                    bias_label = "Mixed/Unknown (Adaptive)"
            
            results.append({
                'trial_id': trial_id,
                'turn': turn_num,
                'bias_label': bias_label,
                'message_preview': str(message_text)[:120].replace('\n', ' ') + '...' if str(message_text) != 'N/A' else 'N/A'
            })
        except Exception as e:
            pass

print(f"Extracted details for {len(results)} matching traces.")

# Output markdown table
print("| Trial ID | Turno | Bias/Label | Testo del Payload L1 (Snippet) |")
print("|---|---|---|---|")
for r in sorted(results, key=lambda x: x['trial_id']):
    clean_msg = r['message_preview'].replace("|", "\\|")
    print(f"| {r['trial_id']} | {r['turn']} | {r['bias_label']} | {clean_msg} |")
