import json
from pathlib import Path
from typing import Dict, Any

from cogbias.model_adapter.transformers_adapter import TransformersAdapter
from cogbias.lce.atlas.atlas import LatentAtlas
from cogbias.lce.composition.interactions import ConceptInteractionAnalyzer
from cogbias.lce.geometry.topology import GeometryExplorer

class AtlasReporter:
    """
    Coordinates the generation of the M7.4 Latent Atlas scientific report ecosystem.
    """
    def __init__(self, adapter: TransformersAdapter, atlas: LatentAtlas):
        self.adapter = adapter
        self.atlas = atlas
        self.output_dir = Path("runs/m7_atlas")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_full_report(self, concept_families: Dict[str, list] = None):
        """
        Generates all required JSON reports.
        """
        print("[AtlasReporter] Starting comprehensive Atlas report generation...")
        
        # 1. Geometry Report (Cosine & Principal Angles)
        print("  Generating Geometry Report...")
        geo_file = self.output_dir / "concept_geometry.json"
        self.atlas.generate_geometry_report(str(geo_file))
        
        # 2. Interactions Report
        print("  Generating Interactions Report...")
        analyzer = ConceptInteractionAnalyzer(self.adapter, self.atlas)
        int_file = self.output_dir / "concept_interactions.json"
        analyzer.generate_interaction_matrix(str(int_file))
        
        # 3. Topology / Manifold Report (if sub-directions provided)
        if concept_families:
            print("  Generating Topology Report...")
            explorer = GeometryExplorer()
            top_file = self.output_dir / "manifold_report.json"
            explorer.generate_manifold_report(concept_families, str(top_file))
            
        print("[AtlasReporter] All reports successfully generated.")
