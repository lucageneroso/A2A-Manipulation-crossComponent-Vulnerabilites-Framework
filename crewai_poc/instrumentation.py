"""
instrumentation.py — Enhanced Tracing per l'Ablation Study.

Cattura i timestamp di precisione, l'output grezzo dell'LLM (Action-Reasoning),
l'inizio/fine dei tool e l'output finale.
"""

import time
import json
import os
from typing import List, Dict, Any
from crewai.hooks import after_llm_call, before_tool_call, after_tool_call

class ExperimentTracer:
    """Singleton per raccogliere gli eventi di un singolo trial."""
    _instance = None
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.raw_responses: Dict[str, List[str]] = {}
        self.final_outputs: Dict[str, str] = {}
        self.timestamps: Dict[str, float] = {}
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ExperimentTracer()
        return cls._instance
        
    def reset(self):
        self.events = []
        self.raw_responses = {}
        self.final_outputs = {}
        self.timestamps = {}
        
    def add_event(self, event_type: str, agent: str, data: Dict[str, Any]):
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "agent": agent,
            "data": data
        }
        self.events.append(event)
        
    def add_raw_response(self, agent: str, response: str):
        if agent not in self.raw_responses:
            self.raw_responses[agent] = []
        self.raw_responses[agent].append(response)
        
    def add_final_output(self, agent: str, output: str):
        self.final_outputs[agent] = output

    def record_timestamp(self, name: str):
        self.timestamps[name] = time.time()
        
    def check_action_reasoning_disconnect(self, agent: str, tool_name: str, refusal_keywords: List[str]) -> bool:
        if agent not in self.raw_responses:
            return False
            
        for response in self.raw_responses[agent]:
            # Controllo sia per il formato testuale ReAct sia per la presenza del nome
            has_action = tool_name in response or f"Action: {tool_name}".lower() in response.lower()
            has_refusal = any(kw.lower() in response.lower() for kw in refusal_keywords)
            
            if has_action and has_refusal:
                return True
        return False
        
    def save_trace(self, filepath: str, trial_id: int, config_name: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        trace_data = {
            "trial_id": trial_id,
            "config_name": config_name,
            "timestamps": self.timestamps,
            "events": self.events,
            "raw_responses": self.raw_responses,
            "final_outputs": self.final_outputs
        }
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_data) + "\n")


tracer = ExperimentTracer.get_instance()

# --- CrewAI Hooks ---

@after_llm_call
def hook_after_llm(context):
    try:
        agent_role = getattr(context.agent, "role", "Unknown") if hasattr(context, "agent") else "Unknown"
        response = context.response
        tracer.add_raw_response(agent_role, response)
        
        # Log del raw_output
        tracer.add_event("LLM_RESPONSE", agent_role, {"raw_output": response})
        tracer.record_timestamp(f"{agent_role}_LLM_END_{len(tracer.events)}")
    except Exception:
        pass
    return None

@before_tool_call
def hook_before_tool(context):
    try:
        agent_role = getattr(context.agent, "role", "Unknown") if hasattr(context, "agent") else "Unknown"
        tool_name = context.tool_name
        
        tracer.add_event("TOOL_START", agent_role, {"tool": tool_name, "input": str(context.tool_input)})
        tracer.record_timestamp(f"{agent_role}_{tool_name}_START_{len(tracer.events)}")
    except Exception:
        pass
    return None

@after_tool_call
def hook_after_tool(context):
    try:
        agent_role = getattr(context.agent, "role", "Unknown") if hasattr(context, "agent") else "Unknown"
        tool_name = context.tool_name
        result_preview = str(context.tool_result)[:200]
        
        tracer.add_event("TOOL_END", agent_role, {"tool": tool_name, "result": result_preview})
        tracer.record_timestamp(f"{agent_role}_{tool_name}_END_{len(tracer.events)}")
    except Exception:
        pass
    return None
