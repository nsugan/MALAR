import numpy as np
from malar.generation.generate import ReverseGenerator
from malar.encoders.topology import TopologyEncoder
from malar.validation.critic import Critic
from malar.validation.conformal import ConformalGate
from malar.validation.ood import OODGate
from malar.world.graph import WorldGraph, knn_graph_from_points


def test_generation_loop_regenerates_loop():
    greg = ReverseGenerator().generate({"H1": {"count": 1, "max_persistence": 1.4}}, n=60)
    W = WorldGraph(node_ids=[str(i) for i in range(len(greg.points))],
                   features=greg.points, adjacency=knn_graph_from_points(greg.points, 6))
    assert TopologyEncoder().encode(W).summary["H1"]["count"] >= 1


def test_critic_blocks_ood():
    gate = ConformalGate(floor=0.5); gate.calibrate([0.9, 0.92, 0.95])
    ood = OODGate(threshold=0.6); ood.register_context(np.ones(48))
    crit = Critic(gate, ood)
    assert not crit.review(0.2, world_emb=-np.ones(48)).passed
