import numpy as np
from malar.fields.value_store import ValueStore
from malar.fields.objectives import ObjectiveSpec
from malar.fields.objective_field import ObjectiveField
from malar.fields.functional_store import FunctionalStore
from malar.fields.functional_field import FunctionalField, FunctionalSpec, region_action_set
from malar.fields.values import diffuse_values
from malar.world.adapters.synthetic import SyntheticAdapter


def test_human_value_sticky():
    vs = ValueStore(); of = ObjectiveField(ObjectiveSpec(key="sens"), vs)
    of.set_human_value("c", 0.95)
    ok, proposal = of.propose_change("c", 0.4)
    assert not ok and proposal is not None
    assert vs.value("c", "sens") == 0.95


def test_region_action_set_union():
    fs = FunctionalStore()
    ff = FunctionalField(FunctionalSpec(dim="r", actions=["a", "b", "c"]), fs)
    ff.learn_affordance("x", "a", True); ff.learn_affordance("x", "b", True)
    ff.learn_affordance("y", "c", True)
    A = region_action_set([{"class": "x"}, {"class": "y"}], [ff])
    assert set(A) == {"a", "b", "c"}


def test_value_diffusion_conserves_mass():
    ad = SyntheticAdapter(n_ticks=1, points_per_tick=30, k=6)
    W = ad.to_world(ad.batches(1)[0])
    seed = np.zeros(W.n_nodes); seed[0] = 1.0
    V = diffuse_values(W, seed, steps=20)
    assert abs(V.sum() - 1.0) < 1e-6
    assert (V > 1e-9).sum() > 1
