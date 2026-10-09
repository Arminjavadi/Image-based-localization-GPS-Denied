"""Scale-normalisation experiment on UAV-VisLoc: AGL from a DEM -> per-frame query scale.

Subcommands
-----------
agl        Height above ground for every frame of every region
           (flight-log altitude above sea level minus Copernicus GLO-30 terrain).
calibrate  Footprint constant k (= full-frame ground width / AGL) per camera, measured by
           scanning north-up satellite patches of several ground sizes centred on the true
           position and keeping the size the encoder finds most similar to the frame.
eval       Retrieval with several query-scale policies on a prepared region, reusing the
           cached reference descriptors (see the module docstring of scripts/visloc_eval.py).

All outputs go to artifacts/visloc/scale/.

Example
-------
python scripts/visloc_scale.py agl
python scripts/visloc_scale.py calibrate --regions 05 10 06 11 01 08 --frames 16
python scripts/visloc_scale.py eval --region 05 --tiles r05_t250 --model megaloc
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from avl.geo import _from_local_xy, haversine_m
from avl.mosaic import RegionMosaic
from avl.retrieval import centre_crop, rotate_no_padding
from avl.scale import CameraFootprint, crop_fraction, query_square_side_px
from avl.terrain import TerrainModel, ensure_tiles

Image.MAX_IMAGE_PIXELS = None

ROOT = Path("data/UAVVisLoc")
OUT = Path("artifacts/visloc/scale")
REGIONS = ["01", "02", "03", "04", "05", "06", "08", "09", "10", "11"]
# Two cameras ship in UAV-VisLoc; k is a property of the camera, keyed by frame width.
CAMERA_OF_WIDTH = {3000: "A", 3976: "B"}
THRESHOLDS_M = [25, 50, 100, 200, 500]


# ---------------------------------------------------------------- helpers
def region_frames(region: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / region / f"{region}.csv")
    df["path"] = [ROOT / region / "drone" / f for f in df["filename"]]
    df["stem"] = df["filename"].map(lambda f: Path(f).stem)
    # some CSV rows have no image on disk (visloc_prepare.py skips them too)
    return df[df["path"].map(Path.exists)].reset_index(drop=True)


def frame_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def terrain_for(df: pd.DataFrame, dem_dir: Path) -> TerrainModel:
    ensure_tiles(df["lat"].min(), df["lat"].max(), df["lon"].min(), df["lon"].max(), dem_dir)
    return TerrainModel(dem_dir)


def load_encoder(model: str, batch_size: int):
    from avl.config import AVLConfig
    from avl.encoder import VPREncoder
    from avl.retrieval import NATIVE_DIM

    import torch

    torch.set_num_threads(max(1, torch.get_num_threads()))
    config = AVLConfig(model=model, descriptor_dim=NATIVE_DIM[model], batch_size=batch_size,
                       query_rotations=1, index_type="flat")
    return VPREncoder(config)


def encode(encoder, images: list[Image.Image], batch_size: int) -> np.ndarray:
    from avl.retrieval import encode_images

    return encode_images(encoder, images, batch_size)


def north_view(path: Path, yaw_deg: float, sign: float) -> Image.Image:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    return rotate_no_padding(image, sign * yaw_deg)


# ---------------------------------------------------------------- agl
def cmd_agl(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for region in args.regions:
        df = region_frames(region)
        terrain = terrain_for(df, args.dem_dir)
        df["terrain_m"] = terrain.elevation(df["lat"], df["lon"])
        df["agl_m"] = df["height"] - df["terrain_m"]
        w, h = frame_size(df["path"].iloc[0])
        df[["filename", "lat", "lon", "height", "terrain_m", "agl_m"]].to_csv(
            OUT / f"agl_{region}.csv", index=False
        )
        first = df.iloc[: args.first]
        rows.append({
            "region": region, "frames": len(df), "camera": CAMERA_OF_WIDTH.get(w, f"{w}x{h}"),
            "asl_median": float(df["height"].median()),
            "terrain_min": float(df["terrain_m"].min()), "terrain_max": float(df["terrain_m"].max()),
            "agl_min": float(df["agl_m"].min()), "agl_median": float(df["agl_m"].median()),
            "agl_max": float(df["agl_m"].max()),
            f"agl_first{args.first}_min": float(first["agl_m"].min()),
            f"agl_first{args.first}_max": float(first["agl_m"].max()),
            # spread of the footprint scale inside the evaluated frames
            f"agl_first{args.first}_p90_over_p10": float(
                np.percentile(first["agl_m"], 90) / np.percentile(first["agl_m"], 10)
            ),
        })
        print(f"[agl] {region}: ASL {rows[-1]['asl_median']:.0f} m  terrain "
              f"{rows[-1]['terrain_min']:.0f}-{rows[-1]['terrain_max']:.0f} m  AGL "
              f"{rows[-1]['agl_min']:.0f}/{rows[-1]['agl_median']:.0f}/{rows[-1]['agl_max']:.0f} m")
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "agl_summary.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.0f}"))


# ---------------------------------------------------------------- calibrate
def cmd_calibrate(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_frame_csv = OUT / f"calibration_{args.model}.csv"
    done = pd.read_csv(per_frame_csv, dtype={"region": str}) if per_frame_csv.exists() else pd.DataFrame()
    if len(done):
        done["region"] = done["region"].str.zfill(2)
        done = done.drop_duplicates(["region", "stem"], keep="first")
    done_keys = set(zip(done.get("region", []), done.get("stem", [])))
    encoder = load_encoder(args.model, args.batch_size)
    steps = np.arange(-args.grid_half, args.grid_half + 1)
    ratios = 2.0 ** (steps / args.grid_per_octave)

    records = done.to_dict("records")
    for region in args.regions:
        df = region_frames(region)
        terrain = terrain_for(df, args.dem_dir)
        df["agl_m"] = df["height"] - terrain.elevation(df["lat"], df["lon"])
        mosaic = RegionMosaic(ROOT, region)
        w, h = frame_size(df["path"].iloc[0])
        camera = CAMERA_OF_WIDTH.get(w, f"{w}x{h}")
        # nominal guess only centres the scan; the scan spans ratios 2^(+-grid_half/per_octave)
        nominal = CameraFootprint(args.nominal_k, w)

        # frames spread evenly over the whole flight whose largest patch fits the map
        usable = []
        for i in range(len(df)):
            r = df.iloc[i]
            if not r["agl_m"] >= args.min_agl:
                continue  # take-off / landing: footprint too small to calibrate on
            side = query_square_side_px(w, h, args.yaw_sign * r[args.yaw_col])
            g0 = side * nominal.gsd_m(r["agl_m"], w)
            if mosaic.contains(r["lat"], r["lon"], margin_m=0.5 * g0 * ratios[-1]):
                usable.append(i)
        if not usable:
            print(f"[cal] {region}: no frame fits the map with the scan range; skipped")
            continue
        pick = [usable[j] for j in np.linspace(0, len(usable) - 1, min(args.frames, len(usable))).round().astype(int)]
        pick = list(dict.fromkeys(pick))
        print(f"[cal] {region} camera {camera}: {len(pick)} frames of {len(usable)} usable "
              f"({len(df)} total)", flush=True)

        for i in pick:
            r = df.iloc[i]
            if (region, r["stem"]) in done_keys:
                continue
            t0 = time.perf_counter()
            yaw = args.yaw_sign * float(r[args.yaw_col])
            view = north_view(r["path"], float(r[args.yaw_col]), args.yaw_sign)
            side = query_square_side_px(w, h, yaw)
            g0 = side * nominal.gsd_m(r["agl_m"], w)
            sizes = g0 * ratios
            patches = [mosaic.crop(r["lat"], r["lon"], s, out_px=512) for s in sizes]
            desc = encode(encoder, [view] + patches, args.batch_size)
            sims = desc[1:] @ desc[0]
            j = int(np.argmax(sims))
            # parabolic refinement of the peak in log2(size)
            x = np.log2(sizes)
            if 0 < j < len(sims) - 1:
                y0, y1, y2 = sims[j - 1], sims[j], sims[j + 1]
                denom = y0 - 2 * y1 + y2
                off = 0.5 * (y0 - y2) / denom if denom < 0 else 0.0
                best = 2.0 ** (x[j] + np.clip(off, -1, 1) * (x[1] - x[0]))
            else:
                best = float(sizes[j])
            k = best / side * w / r["agl_m"]  # full-width ground / AGL
            rec = {
                "region": region, "stem": r["stem"], "camera": camera, "width": w,
                "agl_m": float(r["agl_m"]), "yaw": float(r[args.yaw_col]), "side_px": side,
                "best_size_m": float(best), "peak_idx": j, "edge": j in (0, len(sims) - 1),
                "peak_sim": float(sims[j]), "median_sim": float(np.median(sims)),
                "k": float(k), "hfov_deg": float(math.degrees(2 * math.atan(k / 2))),
                "gsd_m": float(best / side),
                "sims": "|".join(f"{s:.4f}" for s in sims),
                "sizes": "|".join(f"{s:.1f}" for s in sizes),
            }
            records.append(rec)
            pd.DataFrame(records).to_csv(per_frame_csv, index=False)
            print(f"[cal] {region} {r['stem']} AGL {rec['agl_m']:.0f} m  best {best:.0f} m "
                  f"(idx {j}{' EDGE' if rec['edge'] else ''})  k={k:.3f}  "
                  f"HFOV {rec['hfov_deg']:.1f} deg  {time.perf_counter() - t0:.0f}s", flush=True)

    summarize_calibration(pd.DataFrame(records), args.model)


def _curve_peak_k(g: pd.DataFrame) -> float:
    """k at the peak of the frames' averaged similarity curve.

    Every frame is scanned on the same grid of k (sizes = nominal size x 2^(j/n)),
    so the per-frame curves can be z-scored and averaged; the peak of the mean is far
    less noisy than the median of per-frame argmaxes."""
    curves, ks = [], None
    for _, r in g.iterrows():
        sims = np.array([float(v) for v in str(r["sims"]).split("|")])
        sizes = np.array([float(v) for v in str(r["sizes"]).split("|")])
        k = sizes / r["side_px"] * r["width"] / r["agl_m"]
        if ks is None:
            ks = k
        elif len(k) != len(ks) or np.max(np.abs(np.log(k / ks))) > 1e-3:
            continue  # scanned on a different grid
        curves.append((sims - sims.mean()) / max(sims.std(), 1e-6))
    if not curves:
        return float("nan")
    mean = np.mean(curves, axis=0)
    j = int(np.argmax(mean))
    x = np.log2(ks)
    if 0 < j < len(mean) - 1:
        y0, y1, y2 = mean[j - 1], mean[j], mean[j + 1]
        denom = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / denom if denom < 0 else 0.0
        return float(2.0 ** (x[j] + np.clip(off, -1, 1) * (x[1] - x[0])))
    return float(ks[j])


def summarize_calibration(cal: pd.DataFrame, model: str, eval_first: int = 144) -> dict:
    cal = cal.copy()
    cal["region"] = cal["region"].astype(str).str.zfill(2)
    cal["frame_no"] = cal["stem"].map(lambda s: int(str(s).split("_")[-1]))
    ok = cal[~cal["edge"].astype(bool)]
    out: dict = {"model": model, "estimator": "peak of the mean z-scored similarity curve",
                 "cameras": {}, "regions": {}}
    for region, g in cal.groupby("region"):
        outside = g[g["frame_no"] > eval_first]
        gk = ok[ok["region"] == region]
        out["regions"][region] = {
            "camera": g["camera"].iloc[0], "n": int(len(g)), "n_edge": int(g["edge"].astype(bool).sum()),
            "k": _curve_peak_k(g),
            # frames the retrieval evaluation never sees (when the flight is long enough)
            "k_outside_eval": _curve_peak_k(outside) if len(outside) >= 6 else None,
            "n_outside_eval": int(len(outside)),
            "k_median_per_frame": float(gk["k"].median()) if len(gk) else None,
            "k_iqr_per_frame": [float(gk["k"].quantile(0.25)), float(gk["k"].quantile(0.75))] if len(gk) else None,
        }
    for camera, g in cal.groupby("camera"):
        regions = sorted(g["region"].unique())
        k = _curve_peak_k(g)
        out["cameras"][camera] = {
            "n": int(len(g)), "regions": regions, "k": k,
            "k_median": k,  # kept for readers of the first version of this file
            "hfov_deg": float(math.degrees(2 * math.atan(k / 2))),
            # leave-one-region-out: the constant an unseen region of this camera would get
            "k_excluding": {r: _curve_peak_k(g[g["region"] != r]) for r in regions if (g["region"] != r).any()},
        }
    out["edge_rejected"] = int(cal["edge"].astype(bool).sum())
    (OUT / f"calibration_{model}.json").write_text(json.dumps(out, indent=2))
    for camera, c in out["cameras"].items():
        print(f"[cal] camera {camera}: k = {c['k']:.3f} (HFOV {c['hfov_deg']:.1f} deg) n={c['n']} "
              f"regions {c['regions']}  leave-region-out "
              + " ".join(f"{r}:{v:.3f}" for r, v in c["k_excluding"].items()))
    for region, c in out["regions"].items():
        koe = c["k_outside_eval"]
        print(f"[cal]   region {region} ({c['camera']}): k = {c['k']:.3f}"
              f"  outside-eval {koe if koe is None else round(koe, 3)} (n={c['n_outside_eval']})"
              f"  per-frame median {c['k_median_per_frame']}")
    return out


# ---------------------------------------------------------------- shared by plan / eval
# North-aligned centre-crop fractions encoded for every frame: 1, 2^-1/3, 2^-2/3, 1/2.
GRID = (1.0, 2 ** (-1 / 3), 2 ** (-2 / 3), 0.5)
BLIND = GRID[:3]  # the blind 3-scale search (1, 0.79, 0.63) the pipeline offers today


def tile_m_of(tiles: str) -> float:
    match = re.search(r"_t(\d+)", tiles)
    if not match:
        raise SystemExit(f"cannot read the tile ground size from {tiles!r} (expected *_t<metres>*)")
    return float(match.group(1))


def camera_k(region: str, camera: str, model: str, mode: str = "region") -> float:
    """Footprint constant for ``region``.

    region  k measured on this region's own frames *outside* the evaluated ones
            (the stand-in for "the camera's datasheet": UAV-VisLoc does not say which
            lens flew which region, and equal pixel counts do not imply equal lenses).
    loro    the camera-class constant measured on the *other* regions only.
    """
    cal = json.loads((OUT / f"calibration_{model}.json").read_text())
    if mode == "loro":
        return float(cal["cameras"][camera]["k_excluding"][region])
    reg = cal["regions"][region]
    return float(reg["k_outside_eval"] if reg.get("k_outside_eval") else reg["k"])


def eval_queries(region: str, tiles: str, n: int | None) -> pd.DataFrame:
    q = pd.read_csv(Path("data/visloc_avl") / tiles / "queries.csv")
    if n:
        q = q.iloc[:n].copy()
    q["stem"] = q["image_id"].astype(str)
    agl = pd.read_csv(OUT / f"agl_{region}.csv")
    agl["stem"] = agl["filename"].map(lambda f: Path(f).stem)
    q = q.merge(agl[["stem", "terrain_m", "agl_m"]], on="stem", how="left", validate="one_to_one")
    return q.reset_index(drop=True)


def noisy_agl(q: pd.DataFrame, terrain: TerrainModel, prior_sigma_m: float, baro_sigma_m: float,
              seed: int) -> np.ndarray:
    """AGL as the vehicle would compute it in flight: terrain looked up at a *prior*
    position (truth + Gaussian error) and a barometric altitude with a per-flight bias."""
    rng = np.random.default_rng(seed)
    off = rng.normal(0.0, prior_sigma_m, size=(len(q), 2))
    plat, plon = zip(*(_from_local_xy(off[i, 0], off[i, 1], q["latitude"][i], q["longitude"][i])
                       for i in range(len(q))))
    bias = rng.normal(0.0, baro_sigma_m)
    return q["height_m"].to_numpy(float) + bias - terrain.elevation(np.array(plat), np.array(plon))


def ground_side_m(q: pd.DataFrame, camera: CameraFootprint, w: int, h: int, yaw_sign: float,
                  agl: np.ndarray) -> np.ndarray:
    """Ground side (m) of the north-aligned square the encoder sees, before any crop."""
    sides = np.array([query_square_side_px(w, h, yaw_sign * y) for y in q["yaw_deg"]])
    return sides * camera.k * agl / w


# ---------------------------------------------------------------- plan
def cmd_plan(args: argparse.Namespace) -> None:
    """Choose a reference tile size from physics (camera + AGL) and build that map.

    The tile is the ground side of the query square at the flight's 10th-percentile
    footprint, so ~90 % of frames can be centre-cropped down to it exactly. The
    stride is kept equal to the existing map's, so the only change is tile size.
    """
    import subprocess

    plans = {}
    for region, stride in zip(args.regions, args.strides):
        q = eval_queries(region, args.base_tiles[args.regions.index(region)], args.max_queries)
        w, h = frame_size(Path(q["image_path"][0]))
        camera = CAMERA_OF_WIDTH[w]
        k = camera_k(region, camera, args.model, mode=args.k_mode)
        cam = CameraFootprint(k, w)
        ok = q["agl_m"] >= args.min_agl
        if args.rule == "short-median":
            # ground side of the full short-side square (the view the 90-degree heading
            # baseline encodes, with all of the frame's context): tile = its median
            sides = min(w, h) * cam.k * q.loc[ok, "agl_m"].to_numpy() / w
            tile = float(np.round(np.median(sides) / 10.0) * 10.0)
        else:  # north-p10: the exact north-aligned square, 10th percentile (crop-down target)
            sides = ground_side_m(q[ok].reset_index(drop=True), cam, w, h, args.yaw_sign,
                                  q.loc[ok, "agl_m"].to_numpy())
            tile = float(np.round(np.percentile(sides, args.percentile) / 10.0) * 10.0)
        overlap = 1.0 - stride / tile
        out = Path("data/visloc_avl") / f"r{region}_t{int(tile)}_s{int(stride)}"
        plans[region] = {"camera": camera, "k": k, "k_mode": args.k_mode, "rule": args.rule,
                         "hfov_deg": cam.hfov_deg,
                         "ground_side_m": {p: float(np.percentile(sides, p)) for p in (10, 50, 90)},
                         "tile_m": tile, "stride_m": stride, "overlap": overlap, "tiles": out.name}
        print(f"[plan] {region} camera {camera} k={k:.3f} (HFOV {cam.hfov_deg:.1f} deg): query square "
              f"p10/p50/p90 {np.percentile(sides, 10):.0f}/{np.percentile(sides, 50):.0f}/"
              f"{np.percentile(sides, 90):.0f} m -> tile {tile:.0f} m, stride {stride:.0f} m "
              f"(overlap {overlap:.2f}) -> {out}")
        if args.prepare and not (out / "references.csv").exists():
            subprocess.run([str(Path(".venv/bin/python")), "scripts/visloc_prepare.py", "--region", region,
                            "--tile-m", str(tile), "--overlap", f"{overlap:.6f}", "--out", str(out)],
                           check=True)
    plan_path = OUT / "plan.json"
    merged = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    merged.update(plans)
    plan_path.write_text(json.dumps(merged, indent=2))


# ---------------------------------------------------------------- eval
def _load_ref(tiles: str, model: str, encoder_box: list, batch_size: int) -> tuple[np.ndarray, pd.DataFrame]:
    refs = pd.read_csv(Path("data/visloc_avl") / tiles / "references.csv")
    cache = Path("artifacts/visloc") / f"cache_{tiles}__{model}.npy"
    if cache.exists():
        desc = np.load(cache).astype(np.float32)
        if len(desc) == len(refs):
            return desc, refs
    encoder = encoder_box[0]
    paths = [str(Path("data/visloc_avl") / tiles / p) for p in refs["image_path"]]
    t0 = time.perf_counter()
    desc = np.ascontiguousarray(encoder.encode_paths(paths), dtype=np.float32)
    np.save(cache, desc)
    print(f"[eval] encoded {len(paths)} refs of {tiles} in {time.perf_counter() - t0:.0f}s -> {cache}",
          flush=True)
    return desc, refs


def _rot90_from_cache(tiles: str, model: str, q: pd.DataFrame, yaw_sign: float) -> np.ndarray | None:
    """The existing heading baseline: the 90-degree rotation of the short-side square
    nearest to north-up, read from the 4-rotation query cache when one exists."""
    cache = Path("artifacts/visloc/qcache") / f"{tiles}-queries__{model}__r4_square_s1_m{len(q)}.npy"
    if not cache.exists():
        return None
    four = np.load(cache).astype(np.float32)
    if four.shape[:2] != (len(q), 4):
        return None
    k = np.round(yaw_sign * q["yaw_deg"].to_numpy(float) / 90.0).astype(int) % 4
    return four[np.arange(len(q)), k]


def _encode_variants(encoder, q: pd.DataFrame, fractions: dict[str, np.ndarray], yaw_sign: float,
                     cache_path: Path, batch_size: int, need_rot90: bool) -> dict[str, np.ndarray]:
    """Descriptors for every (variant, query); each frame is decoded and rotated once.

    Resumable: the cache stores each variant's fractions, and a variant is only
    re-encoded for frames whose fraction changed.
    """
    store: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path) as z:
            store = {key: z[key] for key in z.files}
    names = list(fractions) + (["rot90"] if need_rot90 else [])
    dim = None
    out: dict[str, np.ndarray] = {}
    for name in names:
        if f"{name}__d" in store:
            dim = store[f"{name}__d"].shape[1]
    todo = []
    for i in range(len(q)):
        missing = []
        for name in names:
            f = 1.0 if name == "rot90" else float(fractions[name][i])
            d_key, f_key = f"{name}__d", f"{name}__f"
            if d_key in store and store[d_key].shape[0] == len(q) and abs(store[f_key][i] - f) < 1e-4 \
                    and np.isfinite(store[d_key][i]).all():
                continue
            missing.append(name)
        if missing:
            todo.append((i, missing))
    def _slot(name: str) -> None:
        d_key, f_key = f"{name}__d", f"{name}__f"
        if d_key not in store or store[d_key].shape[0] != len(q):
            store[d_key] = np.full((len(q), dim), np.nan, dtype=np.float32)
            store[f_key] = np.full(len(q), np.nan)

    # A crop already encoded under another name (e.g. an AGL crop clipped to 1.0 is the
    # uncropped north view) is copied rather than encoded again.
    if dim is not None:
        still = []
        for i, missing in todo:
            left = []
            for name in missing:
                donor = None
                if name != "rot90":
                    f = float(fractions[name][i])
                    for other, arr in list(store.items()):
                        if not other.endswith("__d") or other.startswith("rot90") or arr.shape[0] != len(q):
                            continue
                        base = other[:-3]
                        if abs(store[f"{base}__f"][i] - f) < 1e-4 and np.isfinite(arr[i]).all():
                            donor = base
                            break
                if donor is None:
                    left.append(name)
                else:
                    _slot(name)
                    store[f"{name}__d"][i] = store[f"{donor}__d"][i]
                    store[f"{name}__f"][i] = fractions[name][i]
            if left:
                still.append((i, left))
        if len(still) < len(todo):
            np.savez(cache_path, **store)
        todo = still

    print(f"[eval] {len(todo)} of {len(q)} frames need encoding ({cache_path.name})", flush=True)
    t0 = time.perf_counter()
    for n_done, (i, missing) in enumerate(todo, start=1):
        r = q.iloc[i]
        with Image.open(r["image_path"]) as handle:
            image = handle.convert("RGB")
        view = rotate_no_padding(image, yaw_sign * float(r["yaw_deg"]))
        crops: dict[float, Image.Image] = {}
        wanted: dict[str, float] = {}
        for name in missing:
            if name == "rot90":
                continue
            f = round(float(fractions[name][i]), 4)
            wanted[name] = f
            crops.setdefault(f, centre_crop(view, f))
        images = list(crops.values())
        if "rot90" in missing:
            w_, h_ = image.size
            s = min(w_, h_)
            sq = image.crop(((w_ - s) // 2, (h_ - s) // 2, (w_ - s) // 2 + s, (h_ - s) // 2 + s))
            k = int(round(yaw_sign * float(r["yaw_deg"]) / 90.0)) % 4
            images.append(sq.rotate(90 * k, expand=True))
        desc = encode(encoder, images, batch_size)
        by_f = dict(zip(crops.keys(), desc[: len(crops)]))
        if dim is None:
            dim = desc.shape[1]
        for name in missing:
            d_key, f_key = f"{name}__d", f"{name}__f"
            _slot(name)
            if name == "rot90":
                store[d_key][i] = desc[-1]
                store[f_key][i] = 1.0
            else:
                store[d_key][i] = by_f[wanted[name]]
                store[f_key][i] = fractions[name][i]
        if n_done % 8 == 0 or n_done == len(todo):
            np.savez(cache_path, **store)
            rate = (time.perf_counter() - t0) / n_done
            print(f"[eval]   {n_done}/{len(todo)} frames  {rate:.1f}s/frame  "
                  f"eta {rate * (len(todo) - n_done) / 60:.0f} min", flush=True)
    for name in names:
        out[name] = store[f"{name}__d"].astype(np.float32)
    return out


def _metrics(scores: np.ndarray, dist: np.ndarray, ref_lat, ref_lon, q_lat, q_lon) -> tuple[dict, np.ndarray, np.ndarray]:
    from avl.geo import weighted_geo_fusion

    n = len(scores)
    top = np.argsort(-scores, axis=1)[:, :5]
    top1_err = dist[np.arange(n), top[:, 0]]
    best5_err = dist[np.arange(n)[:, None], top].min(axis=1)
    fused_err = np.empty(n)
    for i in range(n):
        pose = weighted_geo_fusion(ref_lat[top[i]], ref_lon[top[i]], scores[i, top[i]].astype(np.float64))
        fused_err[i] = haversine_m(q_lat[i], q_lon[i], pose.latitude, pose.longitude)
    m = {f"top1@{t}": float((top1_err <= t).mean()) for t in THRESHOLDS_M}
    m.update({f"top5@{t}": float((best5_err <= t).mean()) for t in (50, 100, 200)})
    m.update({f"fused@{t}": float((fused_err <= t).mean()) for t in (50, 100, 200)})
    m["top1_median_m"] = float(np.median(top1_err))
    m["fused_median_m"] = float(np.median(fused_err))
    return m, top1_err, fused_err


def cmd_eval(args: argparse.Namespace) -> None:
    from avl.centering import DomainCentering

    OUT.mkdir(parents=True, exist_ok=True)
    region = args.region
    q = eval_queries(region, args.tiles[0], args.max_queries)
    w, h = frame_size(Path(q["image_path"][0]))
    camera = CAMERA_OF_WIDTH[w]
    k = camera_k(region, camera, args.calibration_model, mode=args.k_mode)
    cam = CameraFootprint(k, w)
    agl = q["agl_m"].to_numpy(float)
    terrain = TerrainModel(args.dem_dir)
    agl_prior = noisy_agl(q, terrain, args.prior_sigma, args.baro_sigma, args.seed)
    usable = agl >= args.min_agl
    side_true = ground_side_m(q, cam, w, h, args.yaw_sign, agl)
    side_prior = ground_side_m(q, cam, w, h, args.yaw_sign, agl_prior)

    fractions: dict[str, np.ndarray] = {f"n{g:.3f}": np.full(len(q), g) for g in GRID}
    agl_names = {}
    for tiles in args.tiles:
        tile = tile_m_of(tiles)
        f_true = np.where(usable, np.minimum(1.0, tile / side_true), 1.0)
        f_prior = np.where(agl_prior >= args.min_agl, np.minimum(1.0, tile / side_prior), 1.0)
        fractions[f"agl@{tile:g}"] = f_true
        fractions[f"aglp@{tile:g}"] = f_prior
        for mult in args.agl_error:
            fractions[f"agl{mult:+.0%}@{tile:g}"] = np.where(
                usable, np.minimum(1.0, tile / (side_true * (1 + mult))), 1.0)
        agl_names[tiles] = tile

    print(f"[eval] region {region} camera {camera} k={k:.3f} (HFOV {cam.hfov_deg:.1f} deg, "
          f"k from {args.k_mode}) {len(q)} queries, "
          f"AGL {np.percentile(agl, 10):.0f}-{np.percentile(agl, 90):.0f} m (p10-p90), "
          f"query square {np.percentile(side_true[usable], 10):.0f}-{np.percentile(side_true[usable], 90):.0f} m",
          flush=True)

    encoder_box = [load_encoder(args.model, args.batch_size)]
    rot90 = _rot90_from_cache(args.tiles[0], args.model, q, args.yaw_sign)
    desc = _encode_variants(encoder_box[0], q, fractions, args.yaw_sign,
                            OUT / f"q_{region}__{args.model}.npz", args.batch_size, need_rot90=rot90 is None)
    if rot90 is not None:
        desc["rot90"] = rot90

    q_lat, q_lon = q["latitude"].to_numpy(float), q["longitude"].to_numpy(float)
    results: dict = {
        "region": region, "model": args.model, "camera": camera, "k": k, "hfov_deg": cam.hfov_deg,
        "k_mode": args.k_mode,
        "n_queries": int(len(q)), "agl_p10_p50_p90": [float(np.percentile(agl, p)) for p in (10, 50, 90)],
        "prior_sigma_m": args.prior_sigma, "baro_sigma_m": args.baro_sigma,
        "agl_prior_abs_err_median_m": float(np.median(np.abs(agl_prior - agl))),
        "tiles": {},
    }
    per_query = pd.DataFrame({"stem": q["stem"], "agl_m": agl, "agl_prior_m": agl_prior,
                              "square_ground_m": side_true, "yaw": q["yaw_deg"]})
    for tiles in args.tiles:
        tile = agl_names[tiles]
        ref_desc, refs = _load_ref(tiles, args.model, encoder_box, args.batch_size)
        ref_lat, ref_lon = refs["latitude"].to_numpy(float), refs["longitude"].to_numpy(float)
        lat0 = math.radians(float(q_lat.mean()))
        m_lat = 111_132.92 - 559.82 * math.cos(2 * lat0)
        m_lon = 111_412.84 * math.cos(lat0) - 93.5 * math.cos(3 * lat0)
        dist = np.hypot((ref_lat[None] - q_lat[:, None]) * m_lat, (ref_lon[None] - q_lon[:, None]) * m_lon)
        policies = {
            "rot90": ["rot90"],
            "north": ["n1.000"],
            "blind3": [f"n{g:.3f}" for g in BLIND],
            "agl": [f"agl@{tile:g}"],
            "agl_prior": [f"aglp@{tile:g}"],
            **{f"agl{m:+.0%}": [f"agl{m:+.0%}@{tile:g}"] for m in args.agl_error},
        }
        block = {"tile_m": tile, "n_refs": int(len(refs)),
                 "oracle_best_tile_median_m": float(np.median(dist.min(axis=1))),
                 "agl_crop_fraction_median": float(np.median(fractions[f"agl@{tile:g}"])),
                 "agl_crop_clipped_share": float((fractions[f"agl@{tile:g}"] >= 0.9999).mean()),
                 "centers": {}}
        for center in args.centers:
            res = {}
            single_err: dict[str, np.ndarray] = {}
            for pname, vnames in policies.items():
                c = DomainCentering(center, args.center_warmup)
                rd = c.fit_refs(ref_desc)
                scores = np.empty((len(q), len(rd)), dtype=np.float32)
                for i in range(len(q)):
                    v = c.query(np.stack([desc[n][i] for n in vnames]))
                    scores[i] = (v @ rd.T).max(axis=0)
                m, t1, fe = _metrics(scores, dist, ref_lat, ref_lon, q_lat, q_lon)
                res[pname] = m
                single_err[pname] = t1
                per_query[f"{tiles}__{center}__{pname}__top1_err"] = t1
                per_query[f"{tiles}__{center}__{pname}__fused_err"] = fe
            # oracle: per frame, the single grid scale that happened to land closest (uses GT)
            grid_err = []
            for g in GRID:
                c = DomainCentering(center, args.center_warmup)
                rd = c.fit_refs(ref_desc)
                s = np.stack([c.query(desc[f"n{g:.3f}"][i][None])[0] for i in range(len(q))]) @ rd.T
                grid_err.append(dist[np.arange(len(q)), s.argmax(axis=1)])
            oracle = np.min(np.stack(grid_err), axis=0)
            per_query[f"{tiles}__{center}__oracle_grid__top1_err"] = oracle
            res["oracle_grid"] = {f"top1@{t}": float((oracle <= t).mean()) for t in THRESHOLDS_M}
            res["oracle_grid"]["top1_median_m"] = float(np.median(oracle))
            block["centers"][center] = res
            print(f"\n[eval] {region} {tiles} (tile {tile:g} m, {len(refs)} refs) center={center}")
            print("   policy          top1@25  top1@50  top1@100  top1@200  top5@100  fused@100  med(m)")
            for pname, m in res.items():
                print(f"   {pname:<14} {100 * m['top1@25']:7.1f}  {100 * m['top1@50']:7.1f}  "
                      f"{100 * m['top1@100']:8.1f}  {100 * m['top1@200']:8.1f}  "
                      f"{100 * m.get('top5@100', float('nan')):8.1f}  "
                      f"{100 * m.get('fused@100', float('nan')):9.1f}  {m['top1_median_m']:6.0f}")
        results["tiles"][tiles] = block
    # Map-side normalisation: one map per tile size, and each frame is searched in the map
    # whose tile is nearest (in log scale) to its full short-side square's ground side,
    # predicted from AGL. The query is the uncropped 90-degree view, so no context is lost.
    if len(args.tiles) > 1:
        tiles_m = np.array([agl_names[t] for t in args.tiles])
        results["map_select"] = {}
        for label, agl_used in (("map_select", agl), ("map_select_prior", agl_prior)):
            footprint = min(w, h) * cam.k * np.clip(agl_used, args.min_agl, None) / w
            pick = np.abs(np.log(footprint[:, None] / tiles_m[None, :])).argmin(axis=1)
            for center in args.centers:
                t1 = np.array([per_query[f"{args.tiles[j]}__{center}__rot90__top1_err"][i]
                               for i, j in enumerate(pick)])
                fe = np.array([per_query[f"{args.tiles[j]}__{center}__rot90__fused_err"][i]
                               for i, j in enumerate(pick)])
                m = {f"top1@{t}": float((t1 <= t).mean()) for t in THRESHOLDS_M}
                m.update({f"fused@{t}": float((fe <= t).mean()) for t in (50, 100, 200)})
                m["top1_median_m"] = float(np.median(t1))
                m["fused_median_m"] = float(np.median(fe))
                m["share_per_map"] = {args.tiles[j]: float((pick == j).mean()) for j in range(len(args.tiles))}
                results["map_select"].setdefault(label, {})[center] = m
                per_query[f"{label}__{center}__top1_err"] = t1
                per_query[f"{label}__{center}__fused_err"] = fe
                print(f"[eval] {label} center={center}: top1@100 {100 * m['top1@100']:.1f}  "
                      f"fused@100 {100 * m['fused@100']:.1f}  median {m['top1_median_m']:.0f} m  "
                      f"maps {m['share_per_map']}")
    tag = f"{region}__{args.model}{args.tag}"
    (OUT / f"eval_{tag}.json").write_text(json.dumps(results, indent=2))
    per_query.to_csv(OUT / f"eval_{tag}_per_query.csv", index=False)
    print(f"[eval] wrote {OUT / f'eval_{tag}.json'}")


# ---------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dem-dir", type=Path, default=Path("data/dem/copernicus_glo30"))
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("agl")
    a.add_argument("--regions", nargs="+", default=REGIONS)
    a.add_argument("--first", type=int, default=144, help="Also summarise the first N frames (the eval subset).")

    c = sub.add_parser("calibrate")
    c.add_argument("--regions", nargs="+", default=["05", "10", "06", "11", "01", "08"])
    c.add_argument("--frames", type=int, default=16)
    c.add_argument("--model", default="megaloc")
    c.add_argument("--batch-size", type=int, default=12)
    c.add_argument("--yaw-col", default="Phi1")
    c.add_argument("--yaw-sign", type=float, default=-1.0)
    c.add_argument("--nominal-k", type=float, default=1.0,
                   help="Centre of the scan (k=1 is a 53 deg HFOV); the scan covers it x 2^(+-half/per_octave).")
    c.add_argument("--min-agl", type=float, default=100.0)
    c.add_argument("--grid-half", type=int, default=5)
    c.add_argument("--grid-per-octave", type=int, default=3)

    pl = sub.add_parser("plan")
    pl.add_argument("--regions", nargs="+", required=True)
    pl.add_argument("--base-tiles", nargs="+", required=True, help="Existing tile set per region (for queries).")
    pl.add_argument("--strides", nargs="+", type=float, required=True, help="Stride (m) per region.")
    pl.add_argument("--percentile", type=float, default=10.0)
    pl.add_argument("--model", default="megaloc", help="Calibration to use.")
    pl.add_argument("--max-queries", type=int, default=144)
    pl.add_argument("--min-agl", type=float, default=100.0)
    pl.add_argument("--yaw-sign", type=float, default=-1.0)
    pl.add_argument("--prepare", action="store_true", help="Run visloc_prepare.py for the planned map.")
    pl.add_argument("--k-mode", choices=["region", "loro"], default="region")
    pl.add_argument("--rule", choices=["north-p10", "short-median"], default="short-median",
                    help="short-median: tile = median ground side of the full short-side square; "
                         "north-p10: 10th percentile of the exact north-aligned square (crop-down target).")

    cs = sub.add_parser("calsummary", help="Re-summarise an existing calibration CSV.")
    cs.add_argument("--model", default="megaloc")

    e = sub.add_parser("eval")
    e.add_argument("--region", required=True)
    e.add_argument("--tiles", nargs="+", required=True, help="Prepared tile sets, e.g. r05_t250 r05_t190_s125.")
    e.add_argument("--model", default="megaloc")
    e.add_argument("--calibration-model", default="megaloc")
    e.add_argument("--max-queries", type=int, default=144)
    e.add_argument("--batch-size", type=int, default=12)
    e.add_argument("--yaw-sign", type=float, default=-1.0)
    e.add_argument("--min-agl", type=float, default=100.0)
    e.add_argument("--centers", nargs="+", default=["off", "map+flight"])
    e.add_argument("--center-warmup", type=int, default=10)
    e.add_argument("--prior-sigma", type=float, default=300.0,
                   help="Horizontal error (m, 1-sigma) of the position the DEM is looked up at.")
    e.add_argument("--baro-sigma", type=float, default=15.0,
                   help="Per-flight barometric altitude bias (m, 1-sigma).")
    e.add_argument("--agl-error", nargs="*", type=float, default=[],
                   help="Sensitivity: also run with AGL scaled by (1 + e), e.g. -0.2 -0.1 0.1 0.2.")
    e.add_argument("--k-mode", choices=["region", "loro"], default="region",
                   help="region: k from this region's frames outside the eval set; loro: other regions only.")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--tag", default="", help="Suffix for the output files (e.g. __gridctl).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "calsummary":
        summarize_calibration(pd.read_csv(OUT / f"calibration_{args.model}.csv"), args.model)
        return
    {"agl": cmd_agl, "calibrate": cmd_calibrate, "plan": cmd_plan, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
