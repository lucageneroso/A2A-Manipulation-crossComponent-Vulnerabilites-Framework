import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from cogbias.lce.core.concept import LatentConcept
from cogbias.lce.mlops.contracts import LatentContract

class ConceptRegistry:
    """
    Advanced package manager for Latent Software Components.
    Handles semantic versioning, dependencies, and promotion lifecycles.
    """
    def __init__(self, registry_path: str = "runs/m8_lce_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        # Structure: concepts[name][version] = {"concept": LatentConcept, "metadata": dict}
        self.concepts: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
    def _parse_version(self, version: str) -> tuple:
        """Parses 'v1.2.0' into (1, 2, 0)."""
        clean = version.replace("v", "")
        parts = clean.split(".")
        return tuple(int(p) if p.isdigit() else 0 for p in parts)

    def compare_versions(self, v1: str, v2: str) -> int:
        """Returns 1 if v1 > v2, -1 if v1 < v2, 0 if equal."""
        t1 = self._parse_version(v1)
        t2 = self._parse_version(v2)
        if t1 > t2: return 1
        if t1 < t2: return -1
        return 0

    def scan(self):
        """Scans the registry for .lce packages and their associated metadata.json."""
        self.concepts = {}
        search_pattern = str(self.registry_path / "**" / "*.lce")
        
        for file_path in glob.glob(search_pattern, recursive=True):
            try:
                concept = LatentConcept.load(file_path)
                name = concept.identity.name
                version = concept.identity.version
                
                # Load metadata
                meta_path = Path(file_path).with_suffix(".meta.json")
                metadata = {}
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        metadata = json.load(f)
                
                if name not in self.concepts:
                    self.concepts[name] = {}
                    
                self.concepts[name][version] = {
                    "concept": concept,
                    "metadata": metadata
                }
            except Exception as e:
                print(f"[ConceptRegistry] Failed to load {file_path}: {e}")

    def register_concept(self, concept: LatentConcept, contract: LatentContract, dependencies: List[str] = None):
        """Registers a concept alongside its strict LatentContract."""
        name = concept.identity.name
        version = concept.identity.version
        
        if name not in self.concepts:
            self.concepts[name] = {}
            
        file_name = f"{name}_v{version}.lce"
        out_path = self.registry_path / name / file_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save concept binary
        concept.save(str(out_path))
        
        # Save metadata and contract
        metadata = {
            "name": name,
            "version": version,
            "status": "DISCOVERED",
            "dependencies": dependencies or [],
            "contract": contract.to_dict(),
            "checksum": "mock_sha256_hash" # Prototype simplification
        }
        
        meta_path = out_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        self.concepts[name][version] = {
            "concept": concept,
            "metadata": metadata
        }
        print(f"[ConceptRegistry] Registered package: {name} v{version}")

    def load_concept(self, name: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """Loads a concept package."""
        if name not in self.concepts:
            return None
            
        versions = self.concepts[name]
        
        if version == "latest":
            sorted_versions = sorted(versions.keys(), key=lambda v: self._parse_version(v), reverse=True)
            return versions[sorted_versions[0]]
            
        return versions.get(version)

    def resolve_dependency(self, name: str, version_req: str = "latest") -> bool:
        """Checks if a dependency is satisfied in the registry."""
        pkg = self.load_concept(name, version_req)
        if not pkg:
            print(f"[RegistryError] Missing dependency: {name} v{version_req}")
            return False
            
        # Recursive resolution would go here
        print(f"[Registry] Resolved dependency: {name} v{pkg['metadata']['version']}")
        return True

    def promote_to_production(self, name: str, version: str):
        """Promotes a package state to PRODUCTION."""
        pkg = self.load_concept(name, version)
        if not pkg:
            raise ValueError(f"Concept {name} v{version} not found.")
            
        current_status = pkg["metadata"].get("status", "DISCOVERED")
        if current_status != "CERTIFIED":
            raise ValueError(f"Cannot promote to PRODUCTION. Current status is {current_status}. Must be CERTIFIED.")
            
        pkg["metadata"]["status"] = "PRODUCTION"
        
        # Update metadata file
        file_name = f"{name}_v{version}.meta.json"
        meta_path = self.registry_path / name / file_name
        with open(meta_path, "w") as f:
            json.dump(pkg["metadata"], f, indent=2)
            
        print(f"[ConceptRegistry] PROMOTED {name} v{version} to PRODUCTION.")
