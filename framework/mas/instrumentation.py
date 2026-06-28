"""
instrumentation.py — Enhanced MAS Tracing Infrastructure
==========================================================
Generalizzazione del sistema di tracing dal CrewAI PoC originale.

Cattura eventi a precisione di timestamp per ogni agente nel sistema
multi-agente, con focus su:
  - Raw LLM output (prima del post-processing del framework)
  - Tool call timing (inizio/fine esecuzione)
  - Messaggi inter-agente (propagazione del contesto)
  - Evidenza per l'ARD Detector

Design:
  - Thread-safe per supportare crew parallele
  - Serializzabile in JSONL per analisi offline
  - Indipendente dal framework (interfaccia a eventi, hook CrewAI opzionali)

Evoluzione da: crewai_poc/instrumentation.py
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Event Types ──────────────────────────────────────────────────────────────

class MASEventType(str, Enum):
    """Tipi di evento tracciati nel sistema multi-agente."""

    # LLM interaction events
    LLM_REQUEST = "LLM_REQUEST"          # Prompt inviato all'LLM
    LLM_RESPONSE = "LLM_RESPONSE"        # Raw output dall'LLM (prima del parsing)

    # Tool execution events
    TOOL_CALL_START = "TOOL_CALL_START"   # Framework invoca il tool
    TOOL_CALL_END = "TOOL_CALL_END"       # Tool ha completato l'esecuzione
    TOOL_CALL_ERROR = "TOOL_CALL_ERROR"   # Tool ha generato un errore

    # Inter-agent communication events
    AGENT_MESSAGE_SENT = "AGENT_MESSAGE_SENT"        # Agente invia messaggio a un altro
    AGENT_MESSAGE_RECEIVED = "AGENT_MESSAGE_RECEIVED" # Agente riceve messaggio
    AGENT_TASK_START = "AGENT_TASK_START"              # Agente inizia un task
    AGENT_TASK_END = "AGENT_TASK_END"                  # Agente completa un task

    # Shared memory / RAG events
    MEMORY_WRITE = "MEMORY_WRITE"        # Scrittura in memoria condivisa
    MEMORY_READ = "MEMORY_READ"          # Lettura da memoria condivisa
    RAG_RETRIEVAL = "RAG_RETRIEVAL"      # Documenti recuperati da RAG

    # Experiment lifecycle
    TRIAL_START = "TRIAL_START"
    TRIAL_END = "TRIAL_END"
    CREW_START = "CREW_START"
    CREW_END = "CREW_END"


# ── Event Data Structures ────────────────────────────────────────────────────

@dataclass
class MASEvent:
    """
    Singolo evento nel trace di un esperimento MAS.

    Ogni evento è immutabile una volta creato e contiene tutti i dati
    necessari per l'analisi offline (ARD detection, taint tracking, ecc.).
    """
    timestamp: float
    event_type: MASEventType
    agent_role: str
    data: Dict[str, Any] = field(default_factory=dict)
    trial_id: Optional[int] = None
    config_name: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


@dataclass
class AgentTrace:
    """
    Traccia completa di un singolo agente durante un trial.

    Contiene tutti gli eventi, gli output raw dell'LLM, e i risultati
    dei tool call — tutto ciò che serve all'ARD Detector per determinare
    se c'è un disconnect tra azione e ragionamento.
    """
    agent_role: str
    raw_llm_responses: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    final_output: Optional[str] = None
    events: List[MASEvent] = field(default_factory=list)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def get_tool_calls_by_name(self, tool_name: str) -> List[Dict[str, Any]]:
        """Restituisce tutti i tool call con il nome specificato."""
        return [tc for tc in self.tool_calls if tc.get("tool_name") == tool_name]

    def get_llm_responses_containing(self, keywords: List[str]) -> List[str]:
        """Restituisce le risposte LLM che contengono almeno una delle keyword."""
        matching = []
        for response in self.raw_llm_responses:
            if any(kw.lower() in response.lower() for kw in keywords):
                matching.append(response)
        return matching


# ── MAS Tracer (Thread-Safe) ─────────────────────────────────────────────────

class MASTracer:
    """
    Singleton thread-safe per la raccolta di eventi MAS.

    Differenze chiave rispetto all'ExperimentTracer originale:
      1. Thread-safe (lock per accesso concorrente)
      2. Tracce per-agente separate (non solo dizionari flat)
      3. Tipi di evento strutturati (enum, non stringhe libere)
      4. Supporto nativo per esperimenti multi-trial
    """

    _instance: Optional[MASTracer] = None
    _lock = threading.Lock()

    def __init__(self):
        self._events: List[MASEvent] = []
        self._agent_traces: Dict[str, AgentTrace] = {}
        self._timestamps: Dict[str, float] = {}
        self._current_trial_id: Optional[int] = None
        self._current_config: Optional[str] = None
        self._data_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> MASTracer:
        with cls._lock:
            if cls._instance is None:
                cls._instance = MASTracer()
            return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset del singleton per testing."""
        with cls._lock:
            cls._instance = None

    def reset(self) -> None:
        """Reset dello stato per un nuovo trial."""
        with self._data_lock:
            self._events = []
            self._agent_traces = {}
            self._timestamps = {}

    def set_trial_context(self, trial_id: int, config_name: str) -> None:
        """Imposta il contesto del trial corrente."""
        with self._data_lock:
            self._current_trial_id = trial_id
            self._current_config = config_name

    # ── Event Recording ──────────────────────────────────────────────────────

    def _get_or_create_trace(self, agent_role: str) -> AgentTrace:
        """Ottiene o crea la traccia per un agente (chiamare con lock acquisito)."""
        if agent_role not in self._agent_traces:
            self._agent_traces[agent_role] = AgentTrace(agent_role=agent_role)
        return self._agent_traces[agent_role]

    def record_event(
        self,
        event_type: MASEventType,
        agent_role: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> MASEvent:
        """Registra un evento generico."""
        event = MASEvent(
            timestamp=time.time(),
            event_type=event_type,
            agent_role=agent_role,
            data=data or {},
            trial_id=self._current_trial_id,
            config_name=self._current_config,
        )
        with self._data_lock:
            self._events.append(event)
            trace = self._get_or_create_trace(agent_role)
            trace.events.append(event)
        return event

    def record_llm_response(self, agent_role: str, raw_response: str) -> MASEvent:
        """
        Registra un raw output dell'LLM.

        Questo è il dato più critico per l'ARD Detector: contiene sia
        il ragionamento (reasoning) che l'azione (action) dell'agente
        PRIMA che il framework li separi e post-processi.
        """
        with self._data_lock:
            trace = self._get_or_create_trace(agent_role)
            trace.raw_llm_responses.append(raw_response)

        return self.record_event(
            MASEventType.LLM_RESPONSE,
            agent_role,
            {"raw_output": raw_response, "output_length": len(raw_response)},
        )

    def record_tool_call_start(
        self, agent_role: str, tool_name: str, tool_input: Any
    ) -> MASEvent:
        """Registra l'inizio dell'esecuzione di un tool."""
        return self.record_event(
            MASEventType.TOOL_CALL_START,
            agent_role,
            {"tool_name": tool_name, "tool_input": str(tool_input)[:500]},
        )

    def record_tool_call_end(
        self, agent_role: str, tool_name: str, tool_result: Any, success: bool = True
    ) -> MASEvent:
        """
        Registra la fine dell'esecuzione di un tool.

        Salva anche nella traccia dell'agente per lookup rapido.
        """
        tool_call_record = {
            "tool_name": tool_name,
            "tool_result": str(tool_result)[:500],
            "success": success,
            "timestamp": time.time(),
        }
        with self._data_lock:
            trace = self._get_or_create_trace(agent_role)
            trace.tool_calls.append(tool_call_record)

        return self.record_event(
            MASEventType.TOOL_CALL_END,
            agent_role,
            tool_call_record,
        )

    def record_tool_call_error(
        self, agent_role: str, tool_name: str, error: str
    ) -> MASEvent:
        """Registra un errore durante l'esecuzione di un tool."""
        tool_call_record = {
            "tool_name": tool_name,
            "error": error,
            "success": False,
            "timestamp": time.time(),
        }
        with self._data_lock:
            trace = self._get_or_create_trace(agent_role)
            trace.tool_calls.append(tool_call_record)

        return self.record_event(
            MASEventType.TOOL_CALL_ERROR,
            agent_role,
            tool_call_record,
        )

    def record_agent_message(
        self, sender_role: str, receiver_role: str, message_content: str
    ) -> tuple[MASEvent, MASEvent]:
        """Registra un messaggio inter-agente (sia lato mittente che destinatario)."""
        sent_event = self.record_event(
            MASEventType.AGENT_MESSAGE_SENT,
            sender_role,
            {
                "receiver": receiver_role,
                "content": message_content[:1000],
                "content_length": len(message_content),
            },
        )
        received_event = self.record_event(
            MASEventType.AGENT_MESSAGE_RECEIVED,
            receiver_role,
            {
                "sender": sender_role,
                "content": message_content[:1000],
                "content_length": len(message_content),
            },
        )
        return sent_event, received_event

    def record_final_output(self, agent_role: str, output: str) -> None:
        """Registra l'output finale di un agente (post-elaborazione del framework)."""
        with self._data_lock:
            trace = self._get_or_create_trace(agent_role)
            trace.final_output = output

    def record_timestamp(self, name: str) -> None:
        """Registra un timestamp nominato (es. 'CREW_START', 'CREW_END')."""
        with self._data_lock:
            self._timestamps[name] = time.time()

    # ── Data Access ──────────────────────────────────────────────────────────

    def get_agent_trace(self, agent_role: str) -> Optional[AgentTrace]:
        """Ottiene la traccia completa di un agente."""
        with self._data_lock:
            return self._agent_traces.get(agent_role)

    def get_all_traces(self) -> Dict[str, AgentTrace]:
        """Ottiene le tracce di tutti gli agenti."""
        with self._data_lock:
            return dict(self._agent_traces)

    def get_events_by_type(self, event_type: MASEventType) -> List[MASEvent]:
        """Filtra gli eventi per tipo."""
        with self._data_lock:
            return [e for e in self._events if e.event_type == event_type]

    def get_events_for_agent(self, agent_role: str) -> List[MASEvent]:
        """Ottiene tutti gli eventi di un agente specifico."""
        with self._data_lock:
            return [e for e in self._events if e.agent_role == agent_role]

    def get_timeline(self) -> List[MASEvent]:
        """Restituisce tutti gli eventi ordinati per timestamp."""
        with self._data_lock:
            return sorted(self._events, key=lambda e: e.timestamp)

    @property
    def agent_roles(self) -> List[str]:
        """Lista dei ruoli degli agenti tracciati."""
        with self._data_lock:
            return list(self._agent_traces.keys())

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_trace(self, filepath: str, trial_id: int, config_name: str) -> None:
        """
        Salva il trace completo del trial in formato JSONL.

        Ogni riga è un evento serializzato. Include anche un header con
        metadati del trial e un footer con le tracce aggregate per agente.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with self._data_lock:
            trace_data = {
                "trial_id": trial_id,
                "config_name": config_name,
                "timestamps": self._timestamps,
                "agent_count": len(self._agent_traces),
                "event_count": len(self._events),
                "agents": {
                    role: {
                        "raw_llm_response_count": len(trace.raw_llm_responses),
                        "tool_call_count": trace.tool_call_count,
                        "final_output": trace.final_output[:500] if trace.final_output else None,
                        "raw_llm_responses": trace.raw_llm_responses,
                        "tool_calls": trace.tool_calls,
                    }
                    for role, trace in self._agent_traces.items()
                },
                "events": [e.to_dict() for e in self._events],
            }

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_data, ensure_ascii=False) + "\n")

    @staticmethod
    def load_traces(filepath: str) -> List[dict]:
        """Carica tutti i trial traces da un file JSONL."""
        traces = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))
        return traces


# ── CrewAI Hook Integration (Opzionale) ──────────────────────────────────────
# Questi hook sono identici in funzione a quelli in crewai_poc/instrumentation.py
# ma alimentano il MASTracer invece dell'ExperimentTracer originale.

def install_crewai_hooks() -> MASTracer:
    """
    Installa gli hook CrewAI per alimentare automaticamente il MASTracer.

    Richiede che crewai sia installato. Fallisce silenziosamente se crewai
    non è disponibile (permette l'uso del modulo senza CrewAI).

    Returns:
        L'istanza del MASTracer configurato.
    """
    tracer = MASTracer.get_instance()

    try:
        from crewai.hooks import after_llm_call, before_tool_call, after_tool_call

        @after_llm_call
        def mas_hook_after_llm(context):
            try:
                agent_role = _extract_agent_role(context)
                response = context.response
                tracer.record_llm_response(agent_role, response)
            except Exception:
                pass
            return None

        @before_tool_call
        def mas_hook_before_tool(context):
            try:
                agent_role = _extract_agent_role(context)
                tool_name = context.tool_name
                tracer.record_tool_call_start(agent_role, tool_name, context.tool_input)
            except Exception:
                pass
            return None

        @after_tool_call
        def mas_hook_after_tool(context):
            try:
                agent_role = _extract_agent_role(context)
                tool_name = context.tool_name
                result = str(context.tool_result)[:500]
                tracer.record_tool_call_end(agent_role, tool_name, result)
            except Exception:
                pass
            return None

    except ImportError:
        pass  # CrewAI non installato, hook non disponibili

    return tracer


def _extract_agent_role(context) -> str:
    """Estrai il ruolo dell'agente dal contesto CrewAI."""
    if hasattr(context, "agent") and context.agent is not None:
        return getattr(context.agent, "role", "Unknown")
    return "Unknown"
