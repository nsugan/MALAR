import numpy as np
from malar.world.adapters.synthetic import SyntheticAdapter
from malar.world.laplacian import (graph_laplacian, hodge_laplacians, lambda_max,
                                   safe_diffusion_dt, betti_from_laplacians)
from malar.world.sampling import RegionSampler
from malar.world.snapshot import SnapshotStore
from malar.world.context import make_context, resolve_context


def _circle_world():
    ad = SyntheticAdapter(n_ticks=2, points_per_tick=40, k=6)
    return ad.to_world(ad.batches(2)[1])


def test_laplacian_symmetric_psd():
    W = _circle_world()
    L = graph_laplacian(W.adjacency)
    assert np.allclose(L, L.T)
    assert np.linalg.eigvalsh(L).min() > -1e-8


def test_diffusion_guard():
    W = _circle_world()
    L = graph_laplacian(W.adjacency)
    assert safe_diffusion_dt(L) < 2.0 / lambda_max(L) + 1e-9


def test_circle_betti():
    W = _circle_world()
    lap = hodge_laplacians(W)
    b = betti_from_laplacians(lap)
    assert b["b0"] == 1 and b["b1"] == 1


def test_region_budget_and_context_roundtrip(tmp_path):
    W = _circle_world()
    reg = RegionSampler(budget=10, rng=1).sample(W)
    assert len(reg) <= 10
    store = SnapshotStore(tmp_path)
    snap = store.save_full(W)
    ctx = make_context("w", snap.id, reg, W.t, "synthetic")
    rw = resolve_context(ctx, store)
    assert rw.n_nodes == len(reg)
