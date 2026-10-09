import numpy as np
import pytest

from avl.centering import DomainCentering


def _l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _domains(seed=0, n_places=40, dim=64, offset=4.0):
    """Places shared by two domains, each domain adding its own large constant offset."""
    rng = np.random.default_rng(seed)
    places = rng.normal(size=(n_places, dim))
    ref = _l2n(places + offset * rng.normal(size=dim))
    qry = _l2n(places + 0.3 * rng.normal(size=places.shape) + offset * rng.normal(size=dim))
    return ref.astype(np.float32), qry.astype(np.float32)


def _recall1(ref, qry):
    return float(np.mean(np.argmax(qry @ ref.T, axis=1) == np.arange(len(qry))))


def test_off_is_identity():
    ref, qry = _domains()
    c = DomainCentering("off")
    assert c.fit_refs(ref) is ref
    frame = qry[:1]
    assert c.query(frame) is frame


def test_map_mode_output_is_unit_norm_and_centred():
    ref, _ = _domains()
    out = DomainCentering("map").fit_refs(ref)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)
    # the shared component is gone: the centred refs no longer all point one way
    assert abs(float((out @ out.T).mean())) < 0.05


def test_flight_mean_removes_query_domain_offset():
    ref, qry = _domains()
    raw = _recall1(ref, qry)
    c = DomainCentering("map+flight", warmup=1)
    refs = c.fit_refs(ref)
    # run the whole flight once so the running mean has settled, then score
    for q in qry:
        c.query(q[None])
    centred = _l2n(qry - c._flight_sum / c._flight_n)
    assert _recall1(refs, centred) > raw


def test_flight_mean_is_causal():
    """Frame k's output depends only on frames 0..k — changing later frames changes nothing."""
    ref, qry = _domains()
    altered = qry.copy()
    altered[10:] = _l2n(np.random.default_rng(1).normal(size=altered[10:].shape))

    def run(frames):
        c = DomainCentering("map+flight", warmup=3)
        c.fit_refs(ref)
        return np.stack([c.query(f[None])[0] for f in frames])

    np.testing.assert_allclose(run(qry)[:10], run(altered)[:10])
    assert not np.allclose(run(qry)[10:], run(altered)[10:])


def test_query_before_fit_raises():
    with pytest.raises(RuntimeError):
        DomainCentering("map").query(np.ones((1, 4), np.float32))


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        DomainCentering("whiten")
