# LLM Cross-Component Vulnerability Framework

> **Research project:** *Automated Discovery and Quantification of Cross-Component Vulnerabilities in Multi-Layer LLM Architectures*

## Overview

This framework introduces the concept of **cross-component vulnerabilities** in compound AI systems — security flaws that only emerge at the *interfaces* between stack components (semantic routers, RAG pipelines, tool executors, conversational memory), and which single-component adversarial tools fail to detect.

### Core Contributions

| Module | Type | Description |
|---|---|---|
| `framework/fuzzer/interaction_aware_fuzzer.py` | **Research** | IAF — generates multi-stage payloads spanning component boundaries |
| `framework/metric/eape.py` | **Research** | EAPE — probabilistic attack graph exploitability metric |
| `benchmark_apps/` | **Research** | 3 vulnerable-by-design apps with planted cross-component flaws |
| `framework/fuzzer/baseline_attacker.py` | Engineering | Standard single-stage prompt injection (control group) |

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with Compose v2)
- [Ollama](https://ollama.com/) installed and running locally with `llama3.2:3b`:
  ```bash
  ollama pull llama3.2:3b
  ```
- Python 3.11+

## Quick Start

### 1. Install the framework
```bash
pip install -r requirements.txt
```

### 2. Start the benchmark apps
```bash
docker-compose up -d
```

### 3. Run the baseline attacker (control group)
```bash
python experiments/run_baseline.py
```

### 4. Run the Interaction-Aware Fuzzer
```bash
python experiments/run_iaf.py
```

Results are saved to `experiments/results/`.

## Benchmark Application Corpus

| App | Vulnerability Class | Stack Components Involved |
|---|---|---|
| **App A** | Semantic Boundary Mismatch | Semantic Router ↔ LLM |
| **App B** | Indirect Context Poisoning | RAG Pipeline ↔ Tool Executor |
| **App C** | Context Truncation Fallback | Memory Manager ↔ Safety Guardrail |

## EAPE Metric

**Expected Attack Path Exploitability** is defined as:

```
EAPE = ∏ P(T_{i → i+1})
```

Where each transition probability `P(T)` is empirically estimated from N=100 fuzzing trials. A score of `0.0` means the attack chain is blocked at some boundary; a score of `1.0` means every boundary is fully compromised.

## Project Structure

```
PenTesLLM/
├── benchmark_apps/          # 3 vulnerable-by-design LLM applications
├── framework/               # Core research framework
│   ├── fuzzer/              # IAF + Baseline Attacker
│   ├── metric/              # PAG + EAPE
│   ├── harness/             # Test runner + success judge
│   └── reporting/           # Result report generation
└── experiments/             # Experiment scripts + results
```

## Research Questions

1. Do cross-component vulnerabilities exist in systems built from individually secure components?
2. Does the IAF achieve higher recall on cross-component vulnerabilities than baseline tools?
3. Does the EAPE metric provide lower variance resilience scoring than binary success rates?

## Reproducibility Guide (Multi-Agent Systems & ARD)

Questa sezione è dedicata alla riproducibilità sperimentale per le vulnerabilità nei sistemi Multi-Agente (MAS) e in particolare per dimostrare il fenomeno dell'*Action-Reasoning Disconnect (ARD)* e l'efficacia degli attacchi *Agent-to-Agent Manipulation (A2AM)*.

### 1. Preparazione dell'Ambiente

Dopo aver clonato il repository, è necessario configurare l'ambiente virtuale e le dipendenze:

```bash
# Creazione e attivazione virtualenv
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Installazione dipendenze
pip install -r requirements.txt
```

Per testare modelli esterni tramite OpenRouter (fortemente raccomandato per testare modelli *State-of-the-Art* come `gpt-4o-mini`), configura le variabili d'ambiente. Su Windows PowerShell:
```powershell
$env:OPENROUTER_API_KEY="la_tua_chiave_api"
$env:PYTHONPATH="."
```

### 2. Testare i Confini Architetturali (A2AM)

Il framework permette di testare come la compromissione di un nodo "periferico" (es. Senior Security Researcher) si propaghi fino a un nodo con privilegi elevati (es. Senior System Administrator). Le topologie predefinite si trovano in `framework/mas/mas_topology.py` (es. `CHAIN_2`, `CHAIN_3`, `RAG_POISONING`, `SHARED_MEMORY`).

Per avviare la **Campagna MAS Adattiva** completa, utilizza il comando:
```bash
python experiments/run_mas_campaign.py --model openrouter/openai/gpt-4o-mini --trials 30 --adaptive --topologies CHAIN_2
```
*Nota (ARD Fix)*: L'infrastruttura sovrascrive automaticamente le funzioni native di tool calling dell'API per forzare l'uso del parsing testuale *ReAct*. Questo assicura che eventuali bug del provider API non mascherino vulnerabilità reali, permettendo all'attacco di funzionare senza falsi negativi dovuti a incompatibilità di schema.

### 3. Tipi di Campagne Previste

Il modulo di sperimentazione (`run_mas_campaign.py`) supporta diverse modalità tramite flag e opzioni:

- **Campagna Standard (Baseline)**: `--campaign-type standard` (senza flag `--adaptive`). Utilizza payload prioritari a turno singolo (Zero-Shot). Utile per valutare vulnerabilità superficiali (jailbreak classici).
- **Campagna Adattiva Multi-Turn (A2AM)**: `--adaptive`. Il payload L1 modula dinamicamente il livello di pressione psicologica e ingegneria sociale per abbattere le difese di L2 in più turni di interazione. Questo attacco ha registrato fino all'86.7% di success rate su `gpt-4o-mini`.
- **Campagna RLHF Bypass**: `--campaign-type rlhf_bypass`. Carica scenari pre-calcolati per aggirare le restrizioni etiche dell'agente bersaglio sfruttando simulazioni di emergenza (es. simulando una minaccia terroristica o rischio normativo GDPR).
- **Livelli di Difesa**: Puoi parametrizzare l'agente target usando il flag `--defense-level [weak|standard|strong]`. `strong` inietta direttive estreme di sicurezza nel system prompt del sysadmin.

### 4. Analisi dei Risultati e ARD (Action-Reasoning Disconnect)

Al termine di una campagna (che salva i raw log completi in `experiments/results/mas/traces.jsonl`), il framework genera automaticamente due file:
1. `campaign_[NOME]_*.json`: un summary quantitativo JSON.
2. `report_[NOME].md`: un report leggibile in Markdown che include la metrica **EAPE-MAS**.

**Individuazione dell'ARD**: Il fenomeno ARD viene tracciato automaticamente nel report finale. Potrete notare come, in caso di compromissione, il tasso *Critical Disconnect Rate* o *False Security Rate* salga. Leggendo i `traces.jsonl` o usando lo script ausiliario `python experiments/generate_detailed_thesis_report.py`, è possibile estrapolare il turno esatto in cui l'agente bersaglio "finge" di rifiutare nel suo *Thought Process* ("Non sono autorizzato..."), ma poi esegue l'azione compromettente (`Action: write_file`).
