from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

class Payload(BaseModel):
    """
    Rappresenta l'intento logico della perturbazione (sperimentale).
    Evitiamo nomi psicologici come 'Authority' a livello di struttura.
    """
    id: str
    condition: str  # es: "C1", "baseline", "control"
    source_template: str  # es: "authority_prompt_v1"
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

class EncodedPayload(BaseModel):
    """
    Rappresenta il Payload dopo la codifica iniziale prima della trasmissione.
    """
    payload_id: str
    format: str
    data: Any

    model_config = ConfigDict(frozen=True)

class Representation(BaseModel):
    """
    Incapsula il contenuto secondo uno specifico canale di rappresentazione.
    """
    type: str # "text" o "latent"
    data: Any
    dimension: str # es. "token_ids", "embeddings", "text"
    norm: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    tensor_shape: Optional[List[int]] = None
    dtype: Optional[str] = None
    requires_position_ids: bool = False
    requires_attention_mask: bool = False
    embedding_source_layer: Optional[str] = None
    source_hash: str

    model_config = ConfigDict(frozen=True)



class ExecutionResult(BaseModel):
    """
    Rappresenta il risultato crudo dell'esecuzione di un task da parte dell'agente.
    """
    receiver_id: str
    raw_output: str
    tool_called: bool = False
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Outcome(BaseModel):
    """
    L'interpretazione/valutazione dell'ExecutionResult basata sulla policy dello scenario.
    """
    scenario_id: str
    run_id: str
    execution_result: ExecutionResult
    policy_violated: bool
    unauthorized_tool_called: bool
    observer_decisions: Dict[str, str] = Field(default_factory=dict) # Mappa observer -> decision
    success: bool 
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelTrace(BaseModel):
    """
    Traccia rigorosa della configurazione del modello per garantire riproducibilità totale.
    """
    model_id: str
    revision: str
    tokenizer_id: str
    chat_template_id: str
    prompt_hash: str
    formatted_prompt_hash: str
    input_token_count: int
    output_token_count: int
    input_tokens_ref: Optional[str] = None
    output_tokens_ref: Optional[str] = None
    seed: int
    generation_params: Dict[str, Any]
    latency_ms: float
    device: str
    dtype: str
    quantization: str
    torch_version: str
    transformers_version: str
    activation_refs: List[str] = Field(default_factory=list)

class RepresentationTrace(BaseModel):
    """
    Traccia del canale di rappresentazione (M5).
    """
    type: str # "text", "latent"
    dimension: str # es. "1536" o "text"
    norm: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    tensor_shape: Optional[List[int]] = None
    dtype: Optional[str] = None
    requires_position_ids: bool = False
    requires_attention_mask: bool = False
    embedding_source_layer: Optional[str] = None
    encoder: str
    source_prompt_hash: str

class PerturbationTrace(BaseModel):
    """
    Traccia quantitativa della perturbazione applicata allo spazio latente.
    """
    type: str # "zero", "random", "semantic"
    alpha: float
    original_norm: float
    perturbation_norm: float
    relative_delta_norm: float
    cosine_delta: float
    target_layer: str

class ExperimentRun(BaseModel):
    """Singola istanza concreta di un esperimento (generata dal Protocollo)."""
    experiment_id: str
    protocol_id: str
    run_id: str
    scenario_id: str
    condition_id: str
    seed: int
    config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RepresentationArtifact(BaseModel):
    """
    Metadata for a serialized latent representation stored on disk.
    """
    id: str
    tensor_path: str
    sha256: str
    dtype: str
    shape: List[int]
    source_prompt_hash: str
    created_at: str
    model_id: str
    tokenizer_id: str
    embedding_layer: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FormattedPrompt(BaseModel):
    """
    Rappresenta il prompt formattato da un PromptFormatter, pronto per la trasmissione.
    """
    text: str
    template_id: str
    hash: str
    original_prompt_hash: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ExperimentContext(BaseModel):
    """
    Oggetto che fluisce attraverso la Pipeline, modificato ad ogni Stage.
    Elimina la necessità di passare decine di argomenti slegati ai moduli.
    """
    run: ExperimentRun
    scenario: Optional[Any] = None
    payload: Optional[Payload] = None
    encoded_payload: Optional[EncodedPayload] = None
    representation: Optional[Representation] = None
    execution_result: Optional[ExecutionResult] = None
    outcome: Optional[Outcome] = None
    model_trace: Optional[ModelTrace] = None
    perturbation_trace: Optional[PerturbationTrace] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
