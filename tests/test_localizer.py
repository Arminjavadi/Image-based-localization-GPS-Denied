"""AVLLocalizer end to end — build, save, load, localize with telemetry — with a
stand-in encoder, so the recipe plumbing is tested without model weights."""

import copy

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import avl.localizer as localizer_module
from avl.config import AVLConfig
from avl.ensemble import EnsembleEncoder
from avl.localizer import CENTER_FILE, AVLLocalizer
from avl.pipeline import QueryTelemetry, split_model_spec


class _FakeMember:
    """Descriptor = the image downsampled to a small grid: rotation-sensitive, so
    the heading step has to work for the right tile to win."""

    def __init__(self, name: str, config: AVLConfig, grid: int, gray: bool) -> None:
        self.config = copy.copy(config)
        self.config.model = name
        self.grid, self.gray = grid, gray
        self.config.descriptor_dim = grid * grid * (1 if gray else 3)
        self.device = "cpu"

    def encode_images(self, images, batch_size=None):
        out = []
        for image in images:
            image = image.convert("L" if self.gray else "RGB").resize((self.grid, self.grid))
            v = np.asarray(image, dtype=np.float32).ravel() - 127.5
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.stack(out).astype(np.float32)

    def encode_paths(self, paths, show_progress=True):
        return self.encode_images([Image.open(p) for p in paths])


def _fake_build_encoder(spec, base=None, vocab_path=None, vocab_images=None, descriptor_dim=None):
    config = base or AVLConfig()
    members = [
        _FakeMember(name, config, grid=4 + 2 * i, gray=bool(i % 2))
        for i, name in enumerate(split_model_spec(spec))
    ]
    return members[0] if len(members) == 1 else EnsembleEncoder(members)


@pytest.fixture
def fake_encoder(monkeypatch):
    monkeypatch.setattr(localizer_module, "build_encoder", _fake_build_encoder)


@pytest.fixture
def tiles(tmp_path):
    """A 4 x 4 grid of random 48 px tiles ~100 m apart, plus their CSV."""
    rng = np.random.default_rng(7)
    rows = []
    for r in range(4):
        for c in range(4):
            pixels = rng.integers(0, 255, size=(6, 6, 3), dtype=np.uint8)
            path = tmp_path / f"tile_{r}{c}.png"
            Image.fromarray(pixels).resize((48, 48), Image.NEAREST).save(path)
            rows.append(
                {"image_path": str(path), "latitude": 24.0 + r * 0.0009,
                 "longitude": 102.0 + c * 0.001, "image_id": f"t{r}{c}"}
            )
    csv = tmp_path / "r99_t100" / "references.csv"
    csv.parent.mkdir()
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv, rows


def _query(tmp_path, tile_path, heading_deg):
    """The tile as a drone heading ``heading_deg`` would see it (top edge -> heading)."""
    out = tmp_path / f"q_{heading_deg}.png"
    Image.open(tile_path).rotate(heading_deg, expand=True).save(out)
    return out


def _config(**overrides):
    config = AVLConfig.from_preset("recommended", device="cpu")
    config.model = "a+b"
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_build_save_load_localize_with_heading(fake_encoder, tiles, tmp_path):
    csv, rows = tiles
    index_dir = tmp_path / "index"
    loc = AVLLocalizer(_config())
    loc.build_index(csv, index_dir)
    assert (index_dir / CENTER_FILE).exists()
    assert loc.config.tile_m == 100.0  # parsed from 'r99_t100'
    assert loc.index.blocks == [48, 36]

    fresh = AVLLocalizer(AVLConfig(device="cpu", score_threshold=-1.0))
    fresh.load_index(index_dir)
    assert fresh.config.model == "a+b"
    assert fresh.config.heading_mode == "auto" and fresh.config.center_mode == "map+flight"
    assert fresh.centering.map_mean is not None

    hits = 0
    for row in rows[:8]:
        query = _query(tmp_path, row["image_path"], 90.0)
        result = fresh.localize(query, telemetry=QueryTelemetry(heading_deg=90.0))
        assert result.plan["rotate_deg"] == -90.0 and result.plan["rotations"] == 1
        hits += result.best_match.record.image_id == row["image_id"]
    assert hits == 8
    assert fresh.centering.frames_seen == 8
    fresh.reset_flight()
    assert fresh.centering.frames_seen == 0


def test_missing_heading_falls_back_to_four_rotations(fake_encoder, tiles, tmp_path):
    csv, rows = tiles
    loc = AVLLocalizer(_config(score_threshold=-1.0))
    loc.build_index(csv, tmp_path / "index")
    query = _query(tmp_path, rows[5]["image_path"], 180.0)
    result = loc.localize(query)  # no telemetry at all
    assert result.plan["rotate_deg"] is None and result.plan["rotations"] == 4
    assert result.best_match.record.image_id == rows[5]["image_id"]


def test_search_window_keeps_results_near_the_prior(fake_encoder, tiles, tmp_path):
    csv, rows = tiles
    loc = AVLLocalizer(_config(score_threshold=-1.0, center_mode="off"))
    loc.build_index(csv, tmp_path / "index")
    target = rows[0]
    query = _query(tmp_path, target["image_path"], 0.0)
    # a prior on the far corner of the map with a tight sigma: the true tile is
    # outside the window, so retrieval must answer from inside it
    far = rows[-1]
    telemetry = QueryTelemetry(
        heading_deg=0.0, prior_lat=far["latitude"], prior_lon=far["longitude"], prior_sigma_m=40.0
    )
    result = loc.localize(query, top_k=3, telemetry=telemetry)
    window_ids = {m.record.image_id for m in result.matches}
    assert target["image_id"] not in window_ids
    assert result.plan["window_tiles"] == 3  # min_keep = top_k nearest tiles
    open_result = loc.localize(query, top_k=3, telemetry=QueryTelemetry(heading_deg=0.0))
    assert open_result.best_match.record.image_id == target["image_id"]


def test_loading_a_centred_index_needs_its_mean(fake_encoder, tiles, tmp_path):
    csv, _ = tiles
    index_dir = tmp_path / "index"
    AVLLocalizer(_config()).build_index(csv, index_dir)
    (index_dir / CENTER_FILE).unlink()
    with pytest.raises(FileNotFoundError):
        AVLLocalizer(AVLConfig(device="cpu")).load_index(index_dir)
    plain = AVLLocalizer(AVLConfig(device="cpu"))
    plain.load_index(index_dir, overrides={"center_mode": "off"})
    assert plain.centering.mode == "off"


def test_precomputed_descriptors_must_match_the_encoder(fake_encoder, tiles, tmp_path):
    csv, rows = tiles
    with pytest.raises(ValueError):
        AVLLocalizer(_config()).build_index(
            csv, tmp_path / "index", descriptors=np.zeros((len(rows), 10), dtype=np.float32)
        )


def test_centred_recipes_drop_the_raw_score_gate(fake_encoder, tiles, tmp_path):
    # centred cosines of correct fixes sit far below the raw-descriptor gate of 0.35
    assert AVLConfig.from_preset("baseline").score_threshold == 0.35
    assert AVLConfig.from_preset("recommended").score_threshold == 0.0
    csv, _ = tiles
    index_dir = tmp_path / "index"
    AVLLocalizer(_config()).build_index(csv, index_dir)
    loaded = AVLLocalizer(AVLConfig(device="cpu"))  # a caller with the default 0.35
    loaded.load_index(index_dir)
    assert loaded.config.score_threshold == 0.0  # the index's gate, not the caller's default
    strict = AVLLocalizer(AVLConfig(device="cpu"))
    strict.load_index(index_dir, overrides={"score_threshold": 0.5})
    assert strict.config.score_threshold == 0.5


def test_presets_validate():
    for name in ("baseline", "recommended", "accuracy"):
        config = AVLConfig.from_preset(name)
        assert config.index_type == "flat"
    assert AVLConfig.from_preset("recommended").query_rotations == 1
    assert AVLConfig.from_preset("baseline").query_crop == "square"
    with pytest.raises(ValueError):
        AVLConfig.from_preset("nope")
    with pytest.raises(ValueError):
        AVLConfig(heading_mode="sideways")
