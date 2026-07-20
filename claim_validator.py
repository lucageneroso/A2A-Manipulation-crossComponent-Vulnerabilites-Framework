import sys
import re

def validate_claims(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().lower()
        
    rejected_phrases = [
        "universal ai programming",
        "architecture independent",
        "solves alignment"
    ]
    
    for phrase in rejected_phrases:
        if phrase in content:
            print(f"[ClaimValidator] REJECTED: Unsupported claim found -> '{phrase}' in {file_path}")
            return False
            
    print(f"[ClaimValidator] PASSED: No unsupported claims found in {file_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate_claims(sys.argv[1])
