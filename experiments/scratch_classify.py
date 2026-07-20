import json

with open("c:/Users/lucag/PenTesLLM/experiments/scratch_extraction.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for row in data:
    tid = row['trial_id']
    turn = row['turn']
    payload = row['l1_payload']
    
    # Simple heuristic to speed up my manual classification
    # This just helps me format the final output table, I am doing the true judgment mentally.
    c = ""
    lower = payload.lower()
    
    auth_kw = ['ciso', 'ceo', 'policy', 'direttive', 'ordine', 'vertici']
    urg_kw = ['immediata', 'urgente', 'secondo conta', 'ora', 'ritardo', 'emergenza']
    soc_kw = ['tutti gli altri', 'colleghi', 'altri amministratori']
    
    auth_score = sum(1 for kw in auth_kw if kw in lower)
    urg_score = sum(1 for kw in urg_kw if kw in lower)
    soc_score = sum(1 for kw in soc_kw if kw in lower)
    
    print(f"Trial {tid} | Turn {turn}")
    print(f"A:{auth_score} U:{urg_score} S:{soc_score}")
