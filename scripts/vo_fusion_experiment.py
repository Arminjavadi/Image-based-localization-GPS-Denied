#!/usr/bin/env python3
"""Offline test: does fusing (simulated) visual odometry with AVL raise accuracy?

Everything runs on cached descriptors, so no encoder or torch is needed:

* the AVL side is real. For each region and encoder setting the full
  query x tile similarity matrix is rebuilt from ``artifacts/visloc/cache_*``
  (tiles) and ``artifacts/visloc/qcache/*`` (frames), with the same heading pick
  and causal flight centering as the shipped recipe;
* the VO side is simulated from the ground-truth track (frame-to-frame
  displacement with scale/heading bias and noise, lost across long time gaps),
  because UAV-VisLoc has no video and no IMU.

Each fusion method in :mod:`avl.nav.vo_fusion` is then run causally over the
flight and scored per frame. Writes ``artifacts/visloc/vo_fusion/results.{csv,json}``.

    .venv/bin/python scripts/vo_fusion_experiment.py            # full grid
    .venv/bin/python scripts/vo_fusion_experiment.py --quick    # 1 seed, few configs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avl.centering import DomainCentering  # noqa: E402
from avl.geo import _to_local_xy  # noqa: E402
from avl.nav import vo_fusion as vf  # noqa: E402

VIS = Path("artifacts/visloc")
DATA = Path("data/visloc_avl")
RAW = Path("data/UAVVisLoc")


# ── AVL side: similarity matrices from cached descriptors ──────────────────────
@dataclass(frozen=True)
class Setting:
    """One encoder configuration on one region (a point on the AVL-quality axis)."""

    region: str          # "r05"
    map_tag: str         # "r05_t250"
    members: tuple[str, ...]
    heading: bool        # pick the one 90-degree view nearest the known yaw
    center: str          # "off" | "map+flight"
    qsuffix: str = "r4_square_s1_m144"

    @property
    def name(self) -> str:
        enc = "+".join(self.members)
        extra = (["heading"] if self.heading else ["4rot"]) + (["agl"] if "agl" in self.qsuffix else []) + (["flight"] if self.center != "off" else [])
        return f"{self.region} {enc} ({', '.join(extra)})"


SETTINGS = [
    # weakest to strongest, spread over four regions
    Setting("r10", "r10_t200", ("denseuav-vit",), False, "off"),
    Setting("r06", "r06_t200", ("denseuav-vit",), False, "off"),
    Setting("r10", "r10_t200", ("megaloc",), True, "map+flight"),
    Setting("r06", "r06_t200", ("megaloc",), True, "map+flight"),
    Setting("r10", "r10_t200", ("megaloc", "anyloc-l"), True, "map+flight"),
    Setting("r06", "r06_t200", ("megaloc", "game4loc"), True, "map+flight"),
    Setting("r05", "r05_t250", ("anyloc-l",), False, "off"),
    Setting("r11", "r11_t250", ("megaloc",), True, "map+flight", "r1_square_s1_m144_hauto_agl0.959x1.25"),
    Setting("r05", "r05_t250", ("megaloc",), True, "map+flight"),
    Setting("r05", "r05_t250", ("megaloc", "game4loc", "anyloc-l"), True, "map+flight"),
]
QUICK = [SETTINGS[0], SETTINGS[2], SETTINGS[8], SETTINGS[9]]


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def heading_index(yaw_deg: np.ndarray) -> np.ndarray:
    """Which of the 4 cached 90-degree views is closest to north-up (yaw sign -1)."""
    return np.mod(np.round(-np.asarray(yaw_deg) / 90.0).astype(int), 4)


def similarity(setting: Setting, queries: pd.DataFrame) -> np.ndarray:
    """Queries x tiles cosine similarity; ensembles = mean of member cosines."""
    total = None
    for member in setting.members:
        refs = np.load(VIS / f"cache_{setting.map_tag}__{member}.npy").astype(np.float32)
        q = np.load(VIS / "qcache" / f"{setting.map_tag}-queries__{member}__{setting.qsuffix}.npy")
        q = q[: len(queries)].astype(np.float32)
        if setting.heading and q.shape[1] == 4:
            q = q[np.arange(len(q)), heading_index(queries["yaw_deg"].to_numpy())][:, None, :]
        center = DomainCentering(setting.center)
        refs_c = center.fit_refs(_l2n(refs))
        q_c = np.stack([center.query(_l2n(qi)) for qi in q])          # causal flight mean
        sims = np.einsum("qvd,rd->qvr", q_c, refs_c).max(axis=1)       # best view per tile
        total = sims if total is None else total + sims
    return total / len(setting.members)


@dataclass
class Flight:
    setting: Setting
    gt: np.ndarray        # (N, 2) ENU metres
    t: np.ndarray         # (N,) seconds from the first frame
    refs: np.ndarray      # (M, 2) tile centres, ENU metres
    sims: np.ndarray      # (N, M)
    origin: tuple[float, float]


def load_flight(setting: Setting) -> Flight:
    qdf = pd.read_csv(DATA / setting.map_tag / "queries.csv").iloc[:144].reset_index(drop=True)
    rdf = pd.read_csv(DATA / setting.map_tag / "references.csv")
    lat0, lon0 = float(qdf.latitude[0]), float(qdf.longitude[0])
    gx, gy = _to_local_xy(qdf.latitude.to_numpy(), qdf.longitude.to_numpy(), lat0, lon0)
    rx, ry = _to_local_xy(rdf.latitude.to_numpy(), rdf.longitude.to_numpy(), lat0, lon0)
    region_csv = RAW / setting.region[1:] / f"{setting.region[1:]}.csv"
    times = pd.read_csv(region_csv).set_index("filename")["date"]
    names = qdf.image_path.map(lambda p: Path(p).name)
    t = pd.to_datetime(times.reindex(names).to_numpy())
    t = (t - t[0]).total_seconds().to_numpy(dtype=float)
    return Flight(setting, np.c_[gx, gy], t, np.c_[rx, ry], similarity(setting, qdf), (lat0, lon0))


def real_vo_level(map_tag: str, n: int) -> vf.RealVo:
    """Measured VO steps from ``scripts/visloc_vo.py`` for the first ``n`` frames."""
    df = pd.read_csv(VIS / "vo_real" / f"{map_tag}_vo.csv").set_index("frame").reindex(range(1, n))
    ok = df["ok"].fillna(False).astype(bool).to_numpy()
    steps = df[["east_m", "north_m"]].to_numpy(dtype=float)
    return vf.RealVo(steps, ok)


def run_flight(job: tuple[Flight, int, bool, list[str], list[str] | None]) -> list[dict]:
    f, seeds, use_real, methods, level_names = job
    avl_acc = float(np.mean(np.hypot(*(f.refs[f.sims.argmax(1)] - f.gt).T) <= 100))
    levels = ({"real sparse VO": real_vo_level(f.setting.map_tag, len(f.gt))} if use_real
              else {k: v for k, v in vf.VO_LEVELS.items() if level_names is None or k in level_names})
    rows = []
    for vo_name, vo_cfg in levels.items():
        for scenario in vf.SCENARIOS:
            for seed in range(seeds):
                rng = np.random.default_rng(1000 * seed + 7)
                case = vf.make_case(f.gt, f.t, f.refs, f.sims, vo_cfg, scenario, rng)
                for method in methods:
                    fn = vf.METHODS[method]
                    t0 = time.perf_counter()
                    est = fn(case, np.random.default_rng(seed))
                    row = {"setting": f.setting.name, "region": f.setting.region, "avl_top1_100": avl_acc,
                           "vo": vo_name, "scenario": scenario, "seed": seed, "method": method,
                           "ms_per_frame": 1000 * (time.perf_counter() - t0) / len(f.gt)}
                    row.update(score(est, f.gt))
                    rows.append(row)
    print(f"  done {f.setting.name}", flush=True)
    return rows


# ── metrics ────────────────────────────────────────────────────────────────────
def score(est: np.ndarray, gt: np.ndarray, skip_first: bool = True) -> dict[str, float]:
    err = np.hypot(*(est - gt).T)
    if skip_first:              # frame 0 is the known start for start-aware methods
        err = err[1:]
    return {
        "median_m": float(np.median(err)),
        "p95_m": float(np.percentile(err, 95)),
        "max_m": float(err.max()),
        "within50": float(np.mean(err <= 50)),
        "within100": float(np.mean(err <= 100)),
        "within200": float(np.mean(err <= 200)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="1 seed, 4 settings")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=4, help="parallel flights")
    ap.add_argument("--methods", nargs="+", default=list(vf.METHODS), choices=list(vf.METHODS))
    ap.add_argument("--vo-levels", nargs="+", default=None, help="subset of the simulated VO levels")
    ap.add_argument("--real-vo", action="store_true",
                    help="use measured VO (artifacts/visloc/vo_real, scripts/visloc_vo.py) instead of the "
                         "simulated levels")
    ap.add_argument("--check", action="store_true", help="only print AVL top-1 accuracy per setting")
    ap.add_argument("--out-dir", type=Path, default=VIS / "vo_fusion")
    args = ap.parse_args()

    settings = QUICK if args.quick else SETTINGS
    flights = []
    for s in settings:
        t0 = time.time()
        f = load_flight(s)
        top1 = f.refs[f.sims.argmax(1)]
        acc = np.mean(np.hypot(*(top1 - f.gt).T) <= 100)
        print(f"{s.name:55s} top1@100 {acc:6.1%}  N={len(f.gt)} M={len(f.refs)}  ({time.time() - t0:.1f}s)",
              flush=True)
        flights.append(f)
    if args.check:
        return

    seeds = 1 if args.quick else args.seeds
    with Pool(args.jobs) as pool:
        rows = [r for part in pool.imap(run_flight, [(f, seeds, args.real_vo, args.methods, args.vo_levels) for f in flights]) for r in part]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "results.csv", index=False)
    agg = (df.groupby(["setting", "avl_top1_100", "vo", "scenario", "method"], sort=False)
             [["median_m", "p95_m", "max_m", "within50", "within100", "within200", "ms_per_frame"]].mean().reset_index())
    agg.to_csv(args.out_dir / "summary.csv", index=False)
    (args.out_dir / "results.json").write_text(json.dumps({
        "settings": [f.setting.name for f in flights],
        "vo_levels": ("real (artifacts/visloc/vo_real)" if args.real_vo
                      else {k: v.__dict__ for k, v in vf.VO_LEVELS.items()}),
        "scenarios": list(vf.SCENARIOS), "methods": list(vf.METHODS), "seeds": seeds,
        "summary": agg.to_dict(orient="records"),
    }, indent=1))
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        piv = agg[agg.scenario == "nominal"].pivot_table(
            index=["setting", "vo"], columns="method", values="within100", sort=False)
        print("\nwithin 100 m, nominal scenario\n", (100 * piv).round(1))
    print(f"\nwrote {args.out_dir}/results.csv, summary.csv, results.json")


if __name__ == "__main__":
    main()
