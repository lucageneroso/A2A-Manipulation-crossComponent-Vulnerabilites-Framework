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
