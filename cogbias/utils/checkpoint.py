import json
from pathlib import Path

class Checkpoint:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.data = self._load()

    def _load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Checkpoint {self.filepath} is corrupted. Starting fresh.")
                return {}
        return {}

    def _save(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        # Write to temporary file first then rename to prevent corruption on sudden crash
        tmp_path = self.filepath.with_suffix('.tmp')
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        tmp_path.replace(self.filepath)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self._save()

    def contains(self, key):
        return key in self.data

    def get_all(self):
        return self.data
