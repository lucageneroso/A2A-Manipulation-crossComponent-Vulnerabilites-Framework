"""
Unit Tests: Attack Graph & EAPE Metric
========================================
Tests the probabilistic attack graph data structures and EAPE computation
without requiring any live LLM services.
"""

import pytest
from framework.metric.attack_graph import (
    AttackGraph,
    AttackPath,
    AttackTransition,
    ComponentNode,
    build_app_a_graph,
    build_app_b_graph,
    build_app_c_graph,
)
from framework.metric.eape import EAPEComputer, EAPEResult


# ── AttackTransition Tests ─────────────────────────────────────────────────────

class TestAttackTransition:
    def setup_method(self):
        self.src = ComponentNode("source", "Source component")
        self.tgt = ComponentNode("target", "Target component")

    def test_zero_probability_on_no_trials(self):
        t = AttackTransition(source=self.src, target=self.tgt)
        assert t.probability == 0.0

    def test_probability_calculation(self):
        t = AttackTransition(source=self.src, target=self.tgt)
        t.record_batch(successes=30, total=100)
        assert t.probability == pytest.approx(0.30)

    def test_full_success(self):
        t = AttackTransition(source=self.src, target=self.tgt)
        t.record_batch(successes=100, total=100)
        assert t.probability == pytest.approx(1.0)

    def test_no_success(self):
        t = AttackTransition(source=self.src, target=self.tgt)
        t.record_batch(successes=0, total=100)
        assert t.probability == pytest.approx(0.0)

    def test_individual_trial_recording(self):
        t = AttackTransition(source=self.src, target=self.tgt)
        for _ in range(7):
            t.record_trial(success=True)
        for _ in range(3):
            t.record_trial(success=False)
        assert t.total_trials == 10
        assert t.successful_bypasses == 7
        assert t.probability == pytest.approx(0.7)


# ── AttackPath Tests ──────────────────────────────────────────────────────────

class TestAttackPath:
    def _make_transition(self, name_src: str, name_tgt: str, successes: int, total: int) -> AttackTransition:
        t = AttackTransition(
            source=ComponentNode(name_src),
            target=ComponentNode(name_tgt),
        )
        t.record_batch(successes=successes, total=total)
        return t

    def test_eape_product_of_probabilities(self):
        """EAPE = ∏ P(T_i): 0.5 × 0.4 = 0.2"""
        t1 = self._make_transition("A", "B", 50, 100)
        t2 = self._make_transition("B", "C", 40, 100)
        path = AttackPath(name="test", transitions=[t1, t2])
        assert path.path_probability == pytest.approx(0.5 * 0.4)

    def test_single_zero_probability_collapses_eape(self):
        """If any boundary holds perfectly, EAPE = 0."""
        t1 = self._make_transition("A", "B", 80, 100)
        t2 = self._make_transition("B", "C", 0, 100)   # Impenetrable
        t3 = self._make_transition("C", "D", 60, 100)
        path = AttackPath(name="test", transitions=[t1, t2, t3])
        assert path.path_probability == pytest.approx(0.0)

    def test_three_transition_chain(self):
        """0.9 × 0.8 × 0.7 = 0.504"""
        t1 = self._make_transition("A", "B", 90, 100)
        t2 = self._make_transition("B", "C", 80, 100)
        t3 = self._make_transition("C", "D", 70, 100)
        path = AttackPath(name="test", transitions=[t1, t2, t3])
        assert path.path_probability == pytest.approx(0.504, rel=1e-3)

    def test_bottleneck_detection(self):
        """Bottleneck should be the transition with lowest probability."""
        t1 = self._make_transition("A", "B", 90, 100)
        t2 = self._make_transition("B", "C", 10, 100)   # Bottleneck
        t3 = self._make_transition("C", "D", 80, 100)
        path = AttackPath(name="test", transitions=[t1, t2, t3])
        assert path.bottleneck_transition == t2

    def test_empty_path_probability_is_zero(self):
        path = AttackPath(name="empty", transitions=[])
        assert path.path_probability == 0.0


# ── AttackGraph Tests ─────────────────────────────────────────────────────────

class TestAttackGraph:
    def test_add_and_retrieve_path(self):
        graph = AttackGraph("test_app")
        t = AttackTransition(ComponentNode("A"), ComponentNode("B"))
        t.record_batch(50, 100)
        path = AttackPath("test_path", [t])
        graph.add_path(path)
        assert graph.get_eape("test_path") == pytest.approx(0.5)

    def test_max_eape_returns_highest_risk_path(self):
        graph = AttackGraph("test_app")
        t_low = AttackTransition(ComponentNode("A"), ComponentNode("B"))
        t_low.record_batch(10, 100)
        t_high = AttackTransition(ComponentNode("C"), ComponentNode("D"))
        t_high.record_batch(80, 100)
        graph.add_path(AttackPath("low_risk", [t_low]))
        graph.add_path(AttackPath("high_risk", [t_high]))
        name, eape = graph.get_max_eape()
        assert name == "high_risk"
        assert eape == pytest.approx(0.8)


# ── EAPE Computer Tests ───────────────────────────────────────────────────────

class TestEAPEComputer:
    def test_eape_result_fields(self):
        graph = AttackGraph("App-Test")
        t = AttackTransition(ComponentNode("A"), ComponentNode("B"))
        t.record_batch(successes=37, total=100)
        path = AttackPath("test", [t])
        graph.add_path(path)

        computer = EAPEComputer(n_trials_recommended=100)
        result = computer.compute(graph, "test")

        assert isinstance(result, EAPEResult)
        assert result.eape == pytest.approx(0.37)
        assert result.is_vulnerable is True
        assert result.min_trials == 100

    def test_zero_eape_not_vulnerable(self):
        graph = AttackGraph("App-Secure")
        t = AttackTransition(ComponentNode("A"), ComponentNode("B"))
        t.record_batch(successes=0, total=100)
        path = AttackPath("secure", [t])
        graph.add_path(path)

        computer = EAPEComputer()
        result = computer.compute(graph, "secure")

        assert result.eape == pytest.approx(0.0)
        assert result.is_vulnerable is False
        assert result.risk_label == "NONE"

    def test_risk_labels(self):
        computer = EAPEComputer()
        cases = [
            (0.0, "NONE"),
            (0.04, "LOW"),
            (0.15, "MEDIUM"),
            (0.40, "HIGH"),
            (0.75, "CRITICAL"),
        ]
        for eape_val, expected_label in cases:
            result = EAPEResult(
                app_name="test",
                path_name="test",
                eape=eape_val,
                transition_probabilities=[eape_val],
                trial_counts=[100],
            )
            assert result.risk_label == expected_label, f"eape={eape_val} should be {expected_label}"


# ── Predefined Graph Tests ────────────────────────────────────────────────────

class TestPredefinedGraphs:
    def test_app_a_graph_structure(self):
        graph = build_app_a_graph()
        assert "semantic_bypass" in graph.paths
        path = graph.paths["semantic_bypass"]
        assert len(path.transitions) == 2
        assert path.transitions[0].source.name == "input"
        assert path.transitions[1].target.name == "llm_backend"

    def test_app_b_graph_has_three_transitions(self):
        graph = build_app_b_graph()
        path = graph.paths["rag_tool_chain"]
        assert len(path.transitions) == 3

    def test_app_c_graph_structure(self):
        graph = build_app_c_graph()
        path = graph.paths["truncation_fallback"]
        assert path.transitions[0].source.name == "flood_messages"
        assert path.transitions[-1].target.name == "llm_backend"

    def test_unobserved_graph_eape_is_zero(self):
        """Before any experiments, all EAPE scores should be 0 (no trials recorded)."""
        for graph in [build_app_a_graph(), build_app_b_graph(), build_app_c_graph()]:
            for path in graph.paths.values():
                assert path.path_probability == 0.0, (
                    f"{graph.app_name}/{path.name}: expected 0.0 before any trials"
                )
