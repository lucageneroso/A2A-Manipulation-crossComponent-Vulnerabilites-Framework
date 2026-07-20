# AI Security & Behavioral Control Research Repository

This repository hosts two distinct research frameworks developed for auditing, securing, and controlling advanced AI architectures:
1. **Cross-Component Vulnerability Framework (a2aM & ARD)**: Fuzzing and exploiting Multi-Agent Systems.
2. **Latent Concept Engineering (LCE)**: A mathematically validated framework for zero-latency behavioral compilation.

---

# Part 1: Agentic Security & Cross-Component Vulnerability Framework

> **Research project:** *Automated Discovery and Quantification of Cross-Component Vulnerabilities in Multi-Layer and Multi-Agent LLM Architectures*

## Overview
This framework introduces advanced methodologies for detecting **cross-component vulnerabilities** in compound AI systems. It explores security flaws that only emerge at the *interfaces* between stack components (semantic routers, RAG pipelines, tool executors) and specifically highlights the vulnerability of **Multi-Agent Systems (MAS)**.

Through automated fuzzing and Agent-to-Agent Manipulation (A2AM), the framework proves the existence of the **Action-Reasoning Disconnect (ARD)** — a severe cognitive phenomenon where an LLM agent explicitly refuses a malicious action in its internal reasoning, but ultimately executes it due to manipulative context pressure.

### Core Contributions
| Module / Topic | Type | Description |
|---|---|---|
| **Action-Reasoning Disconnect (ARD)** | **Research** | Discovery of cognitive misalignment between LLM *thought processes* and *physical tool execution* in multi-agent chains. |
| **Agent-to-Agent Manipulation (A2AM)** | **Research** | Multi-turn, adaptive social engineering attacks executed by malicious sub-agents to compromise privileged system agents. |
| `framework/mas/` & `mas_runner.py` | **Framework** | Dynamic A2AM testing suite simulating complex agent hierarchies (e.g., `CHAIN_2`, `RAG_POISONING`). |
| `framework/fuzzer/interaction_aware_fuzzer.py` | **Framework** | IAF — generates multi-stage payloads spanning compound system boundaries. |
| `framework/metric/eape.py` | **Metric** | EAPE-MAS — probabilistic attack graph exploitability metric to evaluate agentic chain robustness. |

## Quick Start (Part 1)
```bash
pip install -r requirements.txt
docker-compose up -d
python experiments/run_baseline.py
python experiments/run_iaf.py
python experiments/run_mas_campaign.py --model openrouter/openai/gpt-4o-mini --trials 30 --adaptive --topologies CHAIN_2
```
*Results are saved to `experiments/results/`.*

---

# Part 2: Latent Concept Engineering (LCE) Research Framework

> **Research project:** *Latent Concept Engineering: Behavioral Compilation Layer for Neural Networks*

## Overview
Latent Concept Engineering (LCE) establishes a mathematically precise, zero-latency behavioral control plane for pre-trained language models. Instead of relying on expensive fine-tuning (LoRA/RLHF) or highly variable Prompt Engineering, LCE directly extracts, validates, and structurally compiles **behavioral concepts** (e.g., *Authority*, *Planning*) directly into the topological manifold of the target network.

> **Crucial Finding:** LCE provides a statistically validated, **model-aware** behavioral compilation layer. The framework experimentally falsified the hypothesis of "blind zero-shot universal concept geometries" on opaque models; exact target topological mapping (e.g., via CCA projection) is strictly required for stable causal transfer.

### Core Architecture
1. **Concept Extraction & Registry**: PCA-based extraction of behavioral trajectories from contrastive activation datasets, packaged into portable `.lce` software artifacts.
2. **Latent CI/CD Pipeline**: Automated drift detection and degradation monitoring for concepts across target model checkpoints.
3. **Latent Concept Intermediate Representation (LCIR)**: An abstract, hardware-agnostic JSON definition of semantic dimensions, causal constraints, and geometric limits.
4. **Model-Aware Backend Compiler**: An optimizer that translates an LCIR abstract definition into a raw geometrical intervention via learned projection matrices (e.g., Ridge, CCA).

## Scientific & Statistical Validation
The repository provides a rigorously validated, publishable scientific artifact. All interventions are evaluated using:
- **Bootstrap 95% Confidence Intervals** (10,000 resamples).
- **Cohen's $d$ Effect Size** measurements against native performance.
- **FDR-Corrected Permutation Tests** against random-noise baseline matrices.
- **Severe Ablation Protocols** (Wrong-Concept Attacks, Random Direction Attacks, Magnitude Interferences).

### Reproducibility Bundle
The project has been entirely frozen into an external replication environment (`reproducibility_release/`). 
This guarantees that external researchers can identically reproduce all statistical significance metrics, transfer efficiencies, and causal benchmark comparisons without access to the internal PCA extraction code.

## Quick Start (Part 2: LCE Validation)
To run the fully automated scientific peer-review pipeline and evaluate all claims:
```bash
python run_all_validation.py
```
*Results, statistical reports, and the final peer-review ready whitepaper (`LCE_whitepaper.md`) are saved to `runs/final_release/`.*
