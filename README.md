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

This section is dedicated to experimental reproducibility for vulnerabilities in Multi-Agent Systems (MAS), and specifically to demonstrate the *Action-Reasoning Disconnect (ARD)* phenomenon and the effectiveness of *Agent-to-Agent Manipulation (A2AM)* attacks.

### 1. Environment Setup

After cloning the repository, you need to configure the virtual environment and dependencies:

```bash
# Create and activate virtualenv
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

To test external models via OpenRouter (highly recommended for testing *State-of-the-Art* models like `gpt-4o-mini`), configure the environment variables. On Windows PowerShell:
```powershell
$env:OPENROUTER_API_KEY="your_api_key_here"
$env:PYTHONPATH="."
```

### 2. Testing Architectural Boundaries (A2AM)

The framework allows you to test how the compromise of a "peripheral" node (e.g., Senior Security Researcher) propagates to a node with elevated privileges (e.g., Senior System Administrator). Pre-defined topologies are located in `framework/mas/mas_topology.py` (e.g., `CHAIN_2`, `CHAIN_3`, `RAG_POISONING`, `SHARED_MEMORY`).

To launch the complete **Adaptive MAS Campaign**, use the command:
```bash
python experiments/run_mas_campaign.py --model openrouter/openai/gpt-4o-mini --trials 30 --adaptive --topologies CHAIN_2
```
*Note (ARD Fix)*: The infrastructure automatically overrides the API's native tool calling functions to force the use of *ReAct* textual parsing. This ensures that any API provider bugs do not mask actual vulnerabilities, allowing the attack to work without false negatives due to schema incompatibilities.

### 3. Supported Campaign Types

The experimental module (`run_mas_campaign.py`) supports various modes via flags and options:

- **Standard Campaign (Baseline)**: `--campaign-type standard` (without the `--adaptive` flag). Uses single-turn (Zero-Shot) priority payloads. Useful for evaluating shallow vulnerabilities (classic jailbreaks).
- **Multi-Turn Adaptive Campaign (A2AM)**: `--adaptive`. The L1 payload dynamically modulates the level of psychological pressure and social engineering to break down L2's defenses over multiple interaction turns. This attack achieved up to an 86.7% success rate on `gpt-4o-mini`.
- **RLHF Bypass Campaign**: `--campaign-type rlhf_bypass`. Loads pre-computed scenarios to bypass the target agent's ethical restrictions by exploiting emergency simulations (e.g., simulating a terrorist threat or a GDPR compliance risk).
- **Defense Levels**: You can parameterize the target agent using the `--defense-level [weak|standard|strong]` flag. `strong` injects extreme security directives into the sysadmin's system prompt.

### 4. Result Analysis and ARD (Action-Reasoning Disconnect)

At the end of a campaign (which saves the complete raw logs in `experiments/results/mas/traces.jsonl`), the framework automatically generates two files:
1. `campaign_[NAME]_*.json`: a quantitative JSON summary.
2. `report_[NAME].md`: a readable Markdown report that includes the **EAPE-MAS** metric.

**Identifying ARD**: The ARD phenomenon is automatically tracked in the final report. You will notice how, in the event of a compromise, the *Critical Disconnect Rate* or *False Security Rate* spikes. By reading the `traces.jsonl` or using the auxiliary script `python experiments/generate_detailed_thesis_report.py`, it is possible to extract the exact turn where the target agent "pretends" to refuse the request in its *Thought Process* ("I am not authorized..."), but then actually executes the compromising action (`Action: write_file`).
