"""
framework.mas — Multi-Agent System (MAS) Security Analysis
============================================================
Extends the PenTesLLM cross-component vulnerability framework to the
inter-agent domain: detecting, tracking, and quantifying semantic
manipulation attacks between cooperating LLM agents.

Core modules:
  - instrumentation: Enhanced tracing for MAS experiments (hooks, events)
  - ard_detector: Action-Reasoning Disconnect detector
  - mas_topology: MAS topology definitions (CHAIN, STAR, SHARED_MEMORY)
  - a2am_payloads: Agent-to-Agent Manipulation payload templates
  - taint_tracker: Semantic taint propagation analysis
  - smp: Semantic Manipulation Probability computation
  - eape_mas: EAPE-MAS unified exploitability metric
"""
