from malar.world.adapters.synthetic import SyntheticAdapter
from malar.core.loop import MalarEngine
from malar.agents.base import AgentContext
from malar.agents.graph import MalarAgentGraph


def test_loop_memory_grows_then_stabilizes():
    ad = SyntheticAdapter(n_ticks=14, points_per_tick=40, k=6)
    eng = MalarEngine(ad, world_id="syn", budget=32)
    sizes = [eng.memory_size() for _ in [eng.step(b) for b in ad.stream()]]
    assert max(sizes) >= 2
    assert sizes[-1] <= max(sizes)  # stabilises (no unbounded growth)


def test_agent_graph_runs_all_stages():
    ad = SyntheticAdapter(n_ticks=3, points_per_tick=40, k=6)
    eng = MalarEngine(ad, world_id="syn", budget=32)
    g = MalarAgentGraph(AgentContext(engine=eng))
    ms = g.run_tick(next(iter(ad.stream())))
    assert ms.policy_decision is not None
    assert ms.curator_decision is not None


def test_review_mode_captures_stage_outputs():
    ad = SyntheticAdapter(n_ticks=1, points_per_tick=40, k=6)
    eng = MalarEngine(ad, world_id="syn", budget=32)
    g = MalarAgentGraph(AgentContext(engine=eng), review_stages={"identification", "memory_write"})
    ms = g.run_tick(next(iter(ad.stream())))
    for s in ["perception", "identification", "policy", "memory_write"]:
        assert s in ms.stage_outputs
