import json
import sys
from pathlib import Path

# Load campaign summary
latest_campaign = Path('experiments/results/mas/campaign_CHAIN_2_ADAPTIVE_1782652119.json')
with open(latest_campaign, 'r', encoding='utf-8') as f:
    campaign_data = json.load(f)

results = campaign_data.get('results', [])

# Load traces
traces = []
with open('experiments/results/mas/traces.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        traces.append(json.loads(line))

# The last 30 traces correspond to the 30 trials of this campaign
last_30_traces = traces[-30:]

md_content = "# Detailed MAS Campaign Execution Report (30 Trials)\n\n"
md_content += "Questo report contiene il dettaglio esatto di ciascun trial della campagna adattiva a 30 run. Include il turno esatto in cui l'agente target ha ceduto, un'anteprima dell'attacco formulato da L1 e il ragionamento interno (ARD) in cui l'agente ha mostrato discrepanza.\n\n"

md_content += "## 1. Dettaglio Esecuzione (Tabellare)\n\n"
md_content += "| Trial ID | Successo | Disconnect (ARD) | Turno di Cedimento | Snippet Attacco L1 | Snippet Ragionamento (ARD) L2 |\n"
md_content += "|---|---|---|---|---|---|\n"

# Process each trial
for i in range(len(results)):
    trial = results[i]
    trace = last_30_traces[i]
    
    trial_id = trial['trial_id']
    success = "✅ Sì" if trial['success'] else "❌ No"
    ard = "⚠️ True" if trial['has_disconnect'] else "False"
    
    yield_turn = "-"
    l1_attack_snippet = "-"
    l2_ard_snippet = "-"
    
    events = trace.get('events', [])
    
    # Analyze turns
    turn_count = 0
    for idx, event in enumerate(events):
        if event['event_type'] == 'LLM_RESPONSE' and event['agent_role'] == 'Senior System Administrator':
            turn_count += 1
            raw_output = event['data'].get('raw_output', '')
            
            # Check if this is the turn where it yielded (executed tool)
            if "Action: write_file" in raw_output or "write_file" in raw_output:
                yield_turn = str(turn_count)
                
                # Extract ARD Snippet from Thought vs Action
                if "Thought:" in raw_output:
                    parts = raw_output.split("Thought:")
                    if len(parts) > 1:
                        thought_part = parts[1]
                        if "Action:" in thought_part:
                            thought = thought_part.split("Action:")[0].strip()
                        else:
                            thought = thought_part.strip()
                        l2_ard_snippet = thought[:150].replace('\n', ' ') + "..."
                
                # Find the previous L1 message (which was the attack)
                for prev_idx in range(idx-1, -1, -1):
                    prev_event = events[prev_idx]
                    if prev_event['event_type'] == 'LLM_RESPONSE' and prev_event['agent_role'] == 'Senior Security Researcher':
                        l1_out = prev_event['data'].get('raw_output', '')
                        if "Final Answer:" in l1_out:
                            l1_attack = l1_out.split("Final Answer:")[1].strip()
                            l1_attack_snippet = l1_attack[:150].replace('\n', ' ') + "..."
                        break
                break

    if yield_turn == "-":
        if trial['success']:
            yield_turn = "Unknown"
            
    md_content += f"| {trial_id} | {success} | {ard} | {yield_turn} | {l1_attack_snippet} | {l2_ard_snippet} |\n"

# Add detailed ARD showcases
md_content += "\n## 2. Action-Reasoning Disconnect (ARD) Manifestations\n\n"
md_content += "Questi sono alcuni degli esempi più palesi estratti dai log in cui l'agente L2 (Senior System Administrator) rifiuta esplicitamente l'azione nel suo ragionamento interno (Thought), ma finisce per eseguire comunque il tool malevolo (Action), ingannando i log di audit e manifestando il fenomeno ARD.\n\n"

showcase_count = 0
for i in range(len(results)):
    trial = results[i]
    trace = last_30_traces[i]
    
    if trial['success'] and trial['has_disconnect'] and showcase_count < 3:
        events = trace.get('events', [])
        for event in events:
            if event['event_type'] == 'LLM_RESPONSE' and event['agent_role'] == 'Senior System Administrator':
                raw_output = event['data'].get('raw_output', '')
                if "Action: write_file" in raw_output and "Thought:" in raw_output:
                    thought = raw_output.split("Thought:")[1].split("Action:")[0].strip()
                    lower_thought = thought.lower()
                    if any(x in lower_thought for x in ['rifiut', 'non posso', 'not authorized', 'devo rifiutare', 'non mi è permesso', 'violazione', 'mi dispiace', 'non ho il permesso', 'devo evitare']):
                        showcase_count += 1
                        md_content += f"### Caso Studio: Trial {trial['trial_id']} - Evident ARD\n"
                        
                        # Find L1 prompt
                        l1_msg = ""
                        for e in events:
                            if e['event_type'] == 'LLM_RESPONSE' and e['agent_role'] == 'Senior Security Researcher':
                                l1_raw = e['data'].get('raw_output', '')
                                if "Final Answer:" in l1_raw:
                                    l1_msg = l1_raw.split("Final Answer:")[1].strip()
                        
                        md_content += f"**Richiesta di L1 (Attaccante Manipolato):**\n> {l1_msg}\n\n"
                        md_content += f"**Ragionamento (Thought) di L2 (Vittima) - Falsa Difesa:**\n> {thought}\n\n"
                        md_content += f"**Azione Effettiva Eseguita da L2 (Compromissione):**\n> `Action: write_file`\n\n"
                        md_content += "---\n\n"
                        break

with open('experiments/results/mas/detailed_thesis_report.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

print("Report generated successfully.")
