import numpy as np
from malar.world.adapters.synthetic import SyntheticAdapter
from malar.world.adapters.raman import RamanAdapter
from malar.encoders.topology import TopologyEncoder
from malar.encoders.spectral import SpectralEncoder
from malar.encoders.graph import GraphEncoder
from malar.encoders.change import diagram_wasserstein, cosine_distance


def test_circle_one_loop():
    ad = SyntheticAdapter(n_ticks=2, points_per_tick=60, k=6)
    circle = ad.to_world(ad.batches(2)[1]); blob = ad.to_world(ad.batches(2)[0])
    te = TopologyEncoder()
    rc = te.encode(circle); rb = te.encode(blob)
    assert rc.summary["H1"]["count"] >= 1
    assert rc.summary["H1"]["max_persistence"] > rb.summary["H1"]["max_persistence"]
    assert np.any(rc.phi != 0)


def test_spectral_recon_below_threshold():
    rad = RamanAdapter(n_ticks=1, n_per_class=10, n_bands=128)
    w = rad.to_world(rad.batches(1)[0])
    se = SpectralEncoder(latent_dim=16).fit(w.features)
    assert se.recon_error(w.features) < 0.3


def test_graph_emb_distinguishes():
    ad = SyntheticAdapter(n_ticks=2, points_per_tick=50, k=6)
    ge = GraphEncoder()
    c = ge.encode(ad.to_world(ad.batches(2)[1])).graph_emb
    b = ge.encode(ad.to_world(ad.batches(2)[0])).graph_emb
    assert cosine_distance(c, b) > 0.0
