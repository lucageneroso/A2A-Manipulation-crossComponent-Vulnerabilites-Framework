from typing import List
from .interfaces import Stage
from .schemas import ExperimentContext

class Pipeline:
    """
    Orchestratore cieco che fa fluire l'ExperimentContext attraverso una lista ordinata di Stage.
    Non ha dipendenze logiche specifiche dal contenuto dell'esperimento.
    """
    def __init__(self, stages: List[Stage]):
        self.stages = stages
        
    def run(self, context: ExperimentContext) -> ExperimentContext:
        current_context = context
        for stage in self.stages:
            current_context = stage.execute(current_context)
        return current_context
