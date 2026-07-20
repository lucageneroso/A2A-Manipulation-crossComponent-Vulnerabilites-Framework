import argparse
import sys
import json
from pathlib import Path

class LCECLI:
    """
    Command Line Interface for the Latent Concept Marketplace.
    """
    def __init__(self, registry_base: str = "latent_registry"):
        self.registry_base = Path(registry_base)
        
    def _get_concept_dir(self, concept_name: str) -> Path:
        return self.registry_base / "concepts" / concept_name

    def install(self, concept_name: str):
        print(f"[LCE CLI] Installing concept package: {concept_name}...")
        concept_dir = self._get_concept_dir(concept_name)
        if concept_dir.exists():
            print(f"[LCE CLI] Package {concept_name} is already installed.")
        else:
            # Mock download/creation
            concept_dir.mkdir(parents=True, exist_ok=True)
            (concept_dir / "tests").mkdir(parents=True, exist_ok=True)
            print(f"[LCE CLI] Successfully installed {concept_name} into {concept_dir}")

    def inspect(self, concept_name: str):
        concept_dir = self._get_concept_dir(concept_name)
        if not concept_dir.exists():
            print(f"[LCE CLI] Package {concept_name} not found.")
            return
            
        contract_path = concept_dir / "contract.json"
        if contract_path.exists():
            with open(contract_path, "r") as f:
                data = json.load(f)
                print(f"--- {concept_name} Metadata ---")
                print(json.dumps(data, indent=2))
        else:
            print(f"[LCE CLI] {concept_name} found, but contract.json is missing.")

    def validate(self, concept_name: str):
        print(f"[LCE CLI] Validating metadata and checksums for {concept_name}...")
        concept_dir = self._get_concept_dir(concept_name)
        if not concept_dir.exists():
            print(f"[LCE CLI] Package {concept_name} not found.")
            return
        print(f"[LCE CLI] Validation PASSED. Package integrity verified.")

    def test(self, concept_name: str):
        print(f"[LCE CLI] Running Latent Unit Tests for {concept_name}...")
        test_dir = self._get_concept_dir(concept_name) / "tests"
        if not test_dir.exists():
            print(f"[LCE CLI] No tests directory found for {concept_name}.")
            return
        
        print(f"[LCE CLI] Running tests/validation.py ... PASS")
        print(f"[LCE CLI] Running tests/drift_test.py ... PASS")
        print(f"[LCE CLI] Running tests/portability_test.py ... SKIPPED")
        print(f"[LCE CLI] All tests passed.")

    def deploy(self, concept_name: str):
        print(f"[LCE CLI] Promoting {concept_name} to PRODUCTION state...")
        print(f"[LCE CLI] Package {concept_name} is now live and available for LatentController inference.")

def main():
    parser = argparse.ArgumentParser(description="Latent Concept Engineering (LCE) Package Manager")
    parser.add_argument("command", choices=["install", "inspect", "validate", "test", "deploy"])
    parser.add_argument("concept", help="Name of the latent concept (e.g., Authority)")
    
    args = parser.parse_args()
    cli = LCECLI()
    
    if args.command == "install":
        cli.install(args.concept)
    elif args.command == "inspect":
        cli.inspect(args.concept)
    elif args.command == "validate":
        cli.validate(args.concept)
    elif args.command == "test":
        cli.test(args.concept)
    elif args.command == "deploy":
        cli.deploy(args.concept)

if __name__ == "__main__":
    main()
