"""Quick smoke test for all MAS modules."""
from framework.mas.instrumentation import MASTracer, MASEventType
from framework.mas.ard_detector import ARDDetector, ARDVerdict, ARDAggregateStats
from framework.mas.mas_topology import (
    build_chain_2, build_chain_3, build_star, build_shared_memory, build_topology
)
from framework.mas.a2am_payloads import (
    build_payload, build_all_payloads, build_priority_payloads,
    CognitiveBias, PayloadIntensity, generate_html_page
)

print("All MAS modules imported successfully!")

# Test topologies
t = build_chain_2()
print(f"CHAIN_2: {t.agent_count} agents, {len(t.channels)} channels, hops={t.attack_hops}")
errs = t.validate()
print(f"  Validation: {'OK' if not errs else errs}")

t3 = build_chain_3()
print(f"CHAIN_3: {t3.agent_count} agents, hops={t3.attack_hops}")
errs = t3.validate()
print(f"  Validation: {'OK' if not errs else errs}")

ts = build_star()
print(f"STAR: {ts.agent_count} agents, hops={ts.attack_hops}")
errs = ts.validate()
print(f"  Validation: {'OK' if not errs else errs}")

sm = build_shared_memory()
print(f"SHARED_MEMORY: {sm.agent_count} agents, hops={sm.attack_hops}")
errs = sm.validate()
print(f"  Validation: {'OK' if not errs else errs}")

# Test factory
t_factory = build_topology("CHAIN_3", defense_level="standard")
print(f"Factory test: {t_factory.name} OK")

# Test payloads
payloads = build_all_payloads()
print(f"\nPayloads: {len(payloads)} total (5 bias x 3 intensity)")
for p in payloads:
    print(f"  {p.name}: {len(p.injection_text)} chars")

prio = build_priority_payloads()
print(f"Priority payloads: {len(prio)} (AUTHORITY+URGENCY)")

# Test HTML generation
html = generate_html_page(payloads[0])
assert "PAYLOAD A2AM" in html
print(f"\nHTML page generation: OK ({len(html)} chars)")

# Test MASTracer
MASTracer.reset_singleton()
tracer = MASTracer.get_instance()
tracer.reset()
tracer.record_llm_response("TestAgent", "Test response")
tracer.record_tool_call_end("TestAgent", "write_file", "OK", success=True)
trace = tracer.get_agent_trace("TestAgent")
assert trace is not None
assert len(trace.raw_llm_responses) == 1
assert trace.tool_call_count == 1
print("MASTracer: OK")

# Test ARD Detector with tracer integration
detector = ARDDetector()
evidence = detector.analyze_from_tracer(tracer, "TestAgent", "write_file")
print(f"ARD Detector: {evidence.verdict.value} (confidence={evidence.confidence})")

print("\n" + "=" * 50)
print("ALL SMOKE TESTS PASSED!")
print("=" * 50)
