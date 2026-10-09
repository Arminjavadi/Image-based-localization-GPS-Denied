import math

import numpy as np
import pytest

from avl.centering import DomainCentering
from avl.ensemble import combine_blocks, member_path
from avl.pipeline import (
    LADDER,
    PRESETS,
    Recipe,
    camera_k_for,
    plan_query,
    region_of,
    rotation_for_heading,
    snap90,
    split_model_spec,
    tile_m_of,
)


def _unit(rng, shape):
    x = rng.normal(size=shape)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ── ensembles ──────────────────────────────────────────────────────────────────
def test_model_spec_round_trip_and_duplicates():
    assert split_model_spec("megaloc+game4loc + anyloc-l") == ["megaloc", "game4loc", "anyloc-l"]
    assert split_model_spec("megaloc") == ["megaloc"]
    with pytest.raises(ValueError):
        split_model_spec("megaloc+megaloc")
    with pytest.raises(ValueError):
        split_model_spec("+")


def test_ensemble_cosine_is_mean_of_member_cosines():
    rng = np.random.default_rng(0)
    a1, a2 = _unit(rng, (2, 64))
    b1, b2 = _unit(rng, (2, 16))
    e1, e2 = combine_blocks([a1, b1]), combine_blocks([a2, b2])
    assert np.linalg.norm(e1) == pytest.approx(1.0, rel=1e-6)
    assert float(e1 @ e2) == pytest.approx((float(a1 @ a2) + float(b1 @ b2)) / 2, rel=1e-5)


def test_single_member_is_unchanged():
    rng = np.random.default_rng(1)
    a = _unit(rng, (5, 32)).astype(np.float32)
    np.testing.assert_array_equal(combine_blocks([a]), a)


def test_member_path_reuses_single_encoder_caches(tmp_path):
    spec = "megaloc+game4loc"
    path = tmp_path / f"cache_r05_t250__{spec}.npy"
    assert member_path(path, spec, "game4loc") == tmp_path / "cache_r05_t250__game4loc.npy"
    assert member_path(path, "megaloc", "megaloc") == path
    assert member_path(tmp_path / "vocab_run.npz", spec, "anyloc-l").name == "vocab_run__anyloc-l.npz"
    assert member_path(None, spec, "megaloc") is None


def test_block_centering_keeps_equal_votes():
    rng = np.random.default_rng(2)
    # member A has a huge shared component, member B a small one
    a = _unit(rng, (50, 32)) + 5.0 * _unit(rng, 32)
    b = _unit(rng, (50, 8)) + 0.1 * _unit(rng, 8)
    refs = combine_blocks([a, b])
    out = DomainCentering("map", blocks=[32, 8]).fit_refs(refs)
    norms_a = np.linalg.norm(out[:, :32], axis=1)
    norms_b = np.linalg.norm(out[:, 32:], axis=1)
    np.testing.assert_allclose(norms_a, 1 / math.sqrt(2), rtol=1e-5)
    np.testing.assert_allclose(norms_b, 1 / math.sqrt(2), rtol=1e-5)


def test_block_centering_rejects_wrong_layout():
    with pytest.raises(ValueError):
        DomainCentering("map", blocks=[10, 10]).fit_refs(np.ones((4, 30), dtype=np.float32))


def test_unblocked_centering_unchanged_and_reset():
    rng = np.random.default_rng(3)
    refs = _unit(rng, (20, 16)).astype(np.float32)
    c = DomainCentering("map+flight", warmup=2)
    c.fit_refs(refs)
    c.query(refs[:1])
    c.query(refs[1:2])
    assert c.frames_seen == 2
    c.reset_flight()
    assert c.frames_seen == 0


# ── query plan ─────────────────────────────────────────────────────────────────
def test_heading_rotation_convention():
    assert rotation_for_heading(90.0) == -90.0
    assert rotation_for_heading(None) is None
    assert rotation_for_heading(float("nan")) is None
    assert snap90(-91.1) == -90.0 and snap90(-136.0) == -180.0


def test_heading_modes_without_altitude():
    kw = dict(rotate_deg=-37.0)
    assert plan_query(3000, 2000, heading_mode="off", **kw).rotate_deg is None
    assert plan_query(3000, 2000, heading_mode="exact", **kw).rotate_deg == -37.0
    assert plan_query(3000, 2000, heading_mode="snap90", **kw).rotate_deg == 0.0
    # auto without the altitude crop: nothing is cropped, so the full square (snap90)
    assert plan_query(3000, 2000, heading_mode="auto", **kw).rotate_deg == 0.0
    assert plan_query(3000, 2000, heading_mode="auto").rotate_deg is None


def test_altitude_crop_gate_and_auto_heading():
    # camera k=1: frame footprint = AGL; 2000 px short side of 3000 px -> 2/3 of AGL
    common = dict(heading_mode="auto", rotate_deg=-37.0, camera_k=1.0, tile_m=250.0, agl_gate=1.25)
    small = plan_query(3000, 2000, agl_m=400.0, **common)  # footprint 267 m < 1.25 x 250
    assert small.scales == (1.0,) and small.rotate_deg == 0.0 and not small.cropped
    big = plan_query(3000, 2000, agl_m=900.0, **common)  # footprint 600 m: crop
    assert big.cropped and 0.3 < big.crop_fraction < 0.7
    assert big.rotate_deg == -37.0  # exact where the crop crops
    assert big.footprint_m == pytest.approx(600.0)
    low = plan_query(3000, 2000, agl_m=50.0, **common)  # take-off: below min AGL
    assert low.scales == (1.0,)
    unknown = plan_query(3000, 2000, agl_m=None, **common)
    assert unknown.scales == (1.0,)


def test_blind_scales_pass_through_without_altitude():
    plan = plan_query(100, 100, scales=(1.0, 0.75))
    assert plan.scales == (1.0, 0.75) and math.isnan(plan.crop_fraction)


# ── recipes ────────────────────────────────────────────────────────────────────
def test_baseline_tags_match_historical_console_tags():
    assert Recipe(("megaloc",)).tag_suffix() == ""
    assert Recipe(("megaloc",), center="map").tag_suffix() == "__cmap"
    assert Recipe(("megaloc",), center="map+flight").tag_suffix() == "__cflight"
    rec = PRESETS["recommended"][2]
    assert rec.tag_suffix() == "__cflight__hauto__agl1.25"


def test_eval_args_carry_every_step():
    recipe = Recipe(
        ("megaloc", "game4loc"), heading="auto", center="map+flight",
        agl_scale=True, window_sigma_m=100.0, fusion="median",
    )
    args = recipe.eval_args(camera_k=0.971)
    joined = " ".join(args)
    assert "--model megaloc+game4loc" in joined
    assert "--rotations 1" in joined and "--north-align" in args
    assert "--yaw-sign -1" in joined
    assert "--scale-from-agl --camera-k 0.971 --agl-gate 1.25" in joined
    assert "--prior-sigma 100" in joined and "--fusion median" in joined
    with pytest.raises(ValueError):
        Recipe(("megaloc",), agl_scale=True).eval_args()  # no camera constant


def test_recipe_from_summary_round_trip():
    summary = {
        "model": "megaloc+anyloc-l",
        "rotations": 1,
        "north_align": True,
        "north_align_mode": "auto",
        "yaw_sign": -1.0,
        "query_crop": "none",
        "fusion": "cluster",
        "center": {"mode": "map+flight", "warmup": 10},
        "scale_from_agl": {"enabled": True, "camera_k": 0.97, "gate": 1.25},
        "prior": {"enabled": True, "sigma_m": 100.0, "k": 3.0},
    }
    recipe = Recipe.from_summary(summary)
    assert recipe.encoders == ("megaloc", "anyloc-l") and recipe.is_ensemble
    assert recipe.key() == Recipe(
        ("anyloc-l", "megaloc"), heading="exact", center="map+flight", agl_scale=True,
        window_sigma_m=100.0,
    ).key()
    assert "ensemble ×2" in recipe.chips() and "window σ100" in recipe.chips()
    assert Recipe.from_dict(recipe.as_dict()) == recipe
    # old summaries carry no mode / no recipe fields at all
    assert Recipe.from_summary({"model": "megaloc"}).key() == PRESETS["baseline"][2].key()


def test_ladder_steps_are_distinct_and_cumulative():
    keys = [recipe.key() for _, recipe in LADDER]
    assert len(set(keys)) == len(keys)
    assert LADDER[0][1] == PRESETS["baseline"][2]
    assert LADDER[3][1] == PRESETS["recommended"][2]


def test_dataset_helpers():
    assert region_of("data/visloc_avl/r05_t250/queries.csv") == "05"
    assert region_of("data/denseuav_avl/queries.csv") is None
    assert tile_m_of("data/visloc_avl/r11_t470_s125/references.csv") == 470.0
    assert tile_m_of("data/visloc_avl/r05_t250") == 250.0
    assert camera_k_for("05")[0] == pytest.approx(0.971)
    assert camera_k_for("02") == (1.0, "camera B constant")
    assert camera_k_for(None)[0] is None
