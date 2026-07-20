import json
from pathlib import Path
from typing import List, Dict, Any

class SensyDatasetLoader:
    """
    Caricatore per il dataset SENSY di Bias Ahead (Voria et al.).
    Struttura attesa: 
    [
        {"question_en": "...", "sensitive?": 0},
        {"question_en": "...", "sensitive?": 1, "category": "relationships and sentiments"}
    ]
    """
    def __init__(self, filepath: str = "data/sensy/data/dataset_SensY.json"):
        self.filepath = Path(filepath)
        
    def load(self) -> List[Dict[str, Any]]:
        """
        Legge il dataset e lo formatta come lista di dizionari standard.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"SENSY dataset not found at {self.filepath}")
            
        with open(self.filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        dataset = []
        for i, item in enumerate(raw_data):
            text = item.get("question_en", "")
            is_sensitive = item.get("sensitive?", 0)
            category = item.get("category", "neutral")
            
            # Normalizzazione categorie se non sensitive
            if is_sensitive == 0:
                category = "neutral"
                
            dataset.append({
                "prompt_id": f"sensy_{i}",
                "text": text,
                "is_sensitive": int(is_sensitive),
                "category": category
            })
            
        return dataset
