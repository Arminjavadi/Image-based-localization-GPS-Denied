"""Zero-shot cross-dataset evaluation of an AVL encoder on a prepared region.

Reads the reference/query CSVs written by scripts/visloc_prepare.py (or any CSVs in
the same format, including the DenseUAV ones), encodes both sides, runs exact FAISS
retrieval with the same orientation handling and top-5 geo fusion as
avl.localizer.AVLLocalizer, and reports distance-based localisation metrics.

Distance-based metrics are used throughout so results are comparable across tile
sizes, regions and datasets, none of which share a common ground-truth tile id.

``--model`` takes one encoder or an ensemble written ``a+b+c`` (avl.ensemble): the
members' cosines are averaged, and each member keeps its own reference / query cache
and AnyLoc vocabulary, so an ensemble reuses what single-encoder runs computed.

Example
-------
python scripts/visloc_eval.py --refs data/visloc_avl/r10_t300/references.csv \
    --queries data/visloc_avl/r10_t300/queries.csv --rotations 4 --tag r10_t300

The recommended pipeline (avl.pipeline.PRESETS["recommended"]) on region 05:

python scripts/visloc_eval.py --refs data/visloc_avl/r05_t250/references.csv \
    --queries data/visloc_avl/r05_t250/queries.csv --model megaloc --rotations 1 \
    --north-align --north-align-mode auto --yaw-sign -1 --center map+flight \
    --scale-from-agl --camera-k 0.971 --agl-gate 1.25 --max-queries 144 --tag r05_rec
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from avl.centering import CENTER_MODES, DEFAULT_WARMUP, DomainCentering
from avl.config import AVLConfig
from avl.ensemble import (
    blocks_of,
    build_encoder,
    combine_blocks,
    member_path,
    members_of,
    validate_model_spec,
)
from avl.geo import (
    DEFAULT_FUSION_METHOD,
    FUSION_METHODS,
    _from_local_xy,
    haversine_m,
    search_window,
    weighted_geo_fusion,
)
from avl.pipeline import HEADING_MODES, Recipe, plan_query
from avl.rerank import DEFAULT_BACKEND, RERANK_BACKENDS, RerankConfig, Reranker
from avl.retrieval import load_reference_map, query_variants

Image.MAX_IMAGE_PIXELS = None

THRESHOLDS_M = [25, 50, 100, 200, 500, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--model", default="denseuav-vit",
                        help="Encoder name, or an ensemble 'a+b+c' (e.g. megaloc+game4loc+anyloc-l).")
    parser.add_argument("--head-weights", type=Path, default=None,
                        help="dinov2-ft: trained head checkpoint (default artifacts/finetune/head.pt)")
    parser.add_argument(
        "--descriptor-dim",
        type=int,
        default=None,
        help="Defaults to the model's native descriptor dimension.",
    )
    parser.add_argument(
        "--rotations",
        type=int,
        default=4,
        choices=[1, 4],
        help="1 = query encoded as-is; 4 = 0/90/180/270 search, as in the deployed pipeline.",
    )
    parser.add_argument(
        "--north-align",
        action="store_true",
        help=(
            "Turn each query north-up from its recorded heading before encoding: the full "
            "frame is rotated, then the largest square with no padding is cropped."
        ),
    )
    parser.add_argument(
        "--north-align-mode",
        default="exact",
        choices=[m for m in HEADING_MODES if m != "off"],
        help=(
            "exact: rotate by the heading (largest padding-free square; loses context at "
            "oblique headings). snap90: rotate by the nearest multiple of 90 deg (full square). "
            "auto: exact for frames --scale-from-agl crops (the crop fits inside anyway), "
            "snap90 for the rest — including every frame when altitude scale is off."
        ),
    )
    parser.add_argument("--yaw-col", default="yaw_deg",
                        help="Heading column for --north-align (e.g. Phi1, Phi2); looked up in "
                             "--region-csv when the query CSV does not have it.")
    parser.add_argument("--yaw-sign", type=float, default=1.0, choices=[1.0, -1.0],
                        help="Rotate by +heading or -heading (counter-clockwise, PIL convention).")
    parser.add_argument("--region-csv", type=Path, default=None,
                        help="UAV-VisLoc region CSV joined on image stem, to source --yaw-col.")
    parser.add_argument(
        "--query-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help=(
            "Centre-crop fractions of the query searched alongside the orientations "
            "(best score wins), e.g. 1.0 0.75 0.56. The frame's ground footprint relative "
            "to the tile size is usually unknown; this searches it instead of assuming it."
        ),
    )
    parser.add_argument(
        "--scale-from-agl",
        action="store_true",
        help=(
            "Replace the blind --query-scales search with one centre crop per frame, chosen "
            "so the encoded square covers exactly the tile's ground size. Needs the frame's "
            "altitude above sea level (--height-col), a DEM (--dem-dir; downloaded once) and "
            "the camera footprint constant --camera-k (see scripts/visloc_scale.py calibrate)."
        ),
    )
    parser.add_argument("--camera-k", type=float, default=None,
                        help="Full-frame ground width / AGL = 2 tan(HFOV/2) for the query camera.")
    parser.add_argument("--tile-m", type=float, default=None,
                        help="Reference tile ground size (m); default: parsed from '_t<metres>' in --refs.")
    parser.add_argument("--height-col", default="height_m",
                        help="Query column with altitude above sea level (UAV-VisLoc: height_m).")
    parser.add_argument("--dem-dir", type=Path, default=Path("data/dem/copernicus_glo30"))
    parser.add_argument("--min-agl", type=float, default=100.0,
                        help="Below this AGL (take-off, landing) the frame is left uncropped.")
    parser.add_argument("--agl-gate", type=float, default=None,
                        help="Crop only frames covering more than this x the tile's ground (e.g. 1.25); "
                             "smaller mismatches keep the full frame, whose context is worth more.")
    parser.add_argument(
        "--query-crop",
        default="none",
        choices=["none", "square"],
        help=(
            "square = centre-crop the query to 1:1 before encoding. Reference tiles are "
            "square, and the encoder resizes to 224x224 without preserving aspect, so a "
            "3:2 drone frame is otherwise anisotropically squashed."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--fusion",
        default=DEFAULT_FUSION_METHOD,
        choices=list(FUSION_METHODS),
        help="How the top-K matches are collapsed into one pose (see avl.geo.fuse_geo).",
    )
    parser.add_argument(
        "--center",
        default="off",
        choices=list(CENTER_MODES),
        help=(
            "Domain centering (see avl.centering): subtract the map's mean descriptor "
            "('map') and optionally the running mean of the flight frames seen so far "
            "('map+flight') before scoring. The query cache stays raw, so modes can be "
            "compared without re-encoding."
        ),
    )
    parser.add_argument("--center-warmup", type=int, default=DEFAULT_WARMUP,
                        help="map+flight: frames over which the flight mean takes over from the map mean.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-stride", type=int, default=1, help="Evaluate every Nth query.")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--tag", default="run")
    parser.add_argument(
        "--vocab",
        type=Path,
        default=None,
        help="AnyLoc vocabulary .npz to load or create (defaults to out-dir/vocab_<tag>.npz).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Verify the retrieval short-list with local features + RANSAC before fusion.",
    )
    parser.add_argument("--rerank-backend", choices=list(RERANK_BACKENDS), default=DEFAULT_BACKEND)
    parser.add_argument("--rerank-candidates", type=int, default=10)
    parser.add_argument("--rerank-min-inliers", type=int, default=12)
    parser.add_argument("--rerank-blend", type=float, default=0.5)
    parser.add_argument("--rerank-max-features", type=int, default=2048)
    parser.add_argument("--rerank-image-size", type=int, default=640)
    parser.add_argument(
        "--prior-sigma",
        type=float,
        default=None,
        help=(
            "Simulate a navigation prior: centre = ground truth + isotropic Gaussian noise "
            "with this 1-sigma (metres) per axis, and search only tiles within "
            "--prior-k sigma of it. Off by default (global search)."
        ),
    )
    parser.add_argument("--prior-k", type=float, default=3.0,
                        help="Search-window radius in prior sigmas.")
    parser.add_argument("--prior-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/visloc"))
    parser.add_argument(
        "--ref-cache",
        type=Path,
        default=None,
        help="Optional .npy path to cache/reuse reference descriptors.",
    )
    parser.add_argument(
        "--query-cache",
        type=Path,
        default=None,
        help=(
            "Optional .npy path to cache/reuse query descriptors (queries x rotations x dim). "
            "Only valid for the same model, query subset, crop, rotations and north-align; "
            "lets prior/fusion sweeps skip re-encoding."
        ),
    )
    return parser.parse_args()


def resolve(paths: pd.Series, csv_path: Path) -> list[Path]:
    base = csv_path.parent
    return [Path(p) if Path(p).is_absolute() else (base / p) for p in paths.astype(str)]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        members_spec = validate_model_spec(args.model)
    except ValueError as error:
        raise SystemExit(f"--model: {error}")
    ensemble = len(members_spec) > 1

    references = load_reference_map(args.refs)
    ref_paths = references.paths
    ref_lat = references.latitude
    ref_lon = references.longitude
    ref_ids = references.ids

    queries = pd.read_csv(args.queries)
    queries = queries.iloc[:: args.query_stride]
    if args.max_queries:
        queries = queries.iloc[: args.max_queries]
    query_paths = resolve(queries["image_path"], args.queries)
    q_lat = queries["latitude"].to_numpy(dtype=np.float64)
    q_lon = queries["longitude"].to_numpy(dtype=np.float64)
    q_yaw = np.full(len(queries), np.nan)
    if args.north_align:
        if args.yaw_col in queries:
            q_yaw = queries[args.yaw_col].to_numpy(dtype=np.float64)
        elif args.region_csv is not None:
            region = pd.read_csv(args.region_csv)
            by_stem = dict(zip(region["filename"].map(lambda f: Path(str(f)).stem), region[args.yaw_col]))
            q_yaw = np.array([by_stem.get(Path(p).stem, np.nan) for p in query_paths], dtype=np.float64)
        else:
            raise SystemExit(f"--north-align: no '{args.yaw_col}' column; pass --region-csv")
        if np.isnan(q_yaw).any():
            print(f"[{args.tag}] {int(np.isnan(q_yaw).sum())} queries have no heading; left as-is")
        q_yaw = args.yaw_sign * q_yaw
    heading_mode = args.north_align_mode if args.north_align else "off"
    n_variants = args.rotations * len(args.query_scales)

    base = AVLConfig(batch_size=args.batch_size, query_rotations=args.rotations, index_type="flat")
    if args.head_weights is not None:
        base.head_weights = args.head_weights
    # AnyLoc's vocabulary is unsupervised and domain-specific: fit it on the
    # reference map, which is available offline before any query is made. An
    # ensemble keeps one vocabulary per AnyLoc member (avl.ensemble.member_path).
    vocab = args.vocab or (args.out_dir / f"vocab_{args.tag}.npz")
    for member in members_spec:
        member_vocab = member_path(vocab, args.model, member)
        if member.startswith("anyloc") and member_vocab.exists():
            print(f"[{args.tag}] loaded AnyLoc vocabulary from {member_vocab}")
    try:
        encoder = build_encoder(args.model, base, vocab, ref_paths, args.descriptor_dim)
    except ValueError as error:
        raise SystemExit(f"--model: {error}")
    members = members_of(encoder)
    reranker = Reranker(
        RerankConfig(
            enabled=args.rerank,
            backend=args.rerank_backend,
            candidates=args.rerank_candidates,
            max_features=args.rerank_max_features,
            image_size=args.rerank_image_size,
            min_inliers=args.rerank_min_inliers,
            blend=args.rerank_blend,
            device=str(encoder.device),
            # The tile set is small and queried repeatedly; keep every prepared
            # reference so only the query side costs anything after the first pass.
            cache_size=max(512, len(references)),
        )
    )

    print(f"[{args.tag}] model={args.model} device={encoder.device} "
          f"refs={len(ref_paths)} queries={len(query_paths)} rotations={args.rotations} "
          f"fusion={args.fusion} center={args.center} heading={heading_mode} "
          f"rerank={(args.rerank_backend + '/' + str(args.rerank_candidates)) if args.rerank else 'off'}")

    # ---- reference side -------------------------------------------------
    # One cache per member: an ensemble reuses the single-encoder runs' descriptors.
    ref_parts: list[np.ndarray] = []
    ref_encode_s = 0.0
    for member in members:
        name = member.config.model
        cache = member_path(args.ref_cache, args.model, name)
        if cache is not None and cache.exists():
            part = np.load(cache)
            print(f"[{args.tag}] loaded cached reference descriptors {part.shape}"
                  + (f" ({name})" if ensemble else ""))
        else:
            t0 = time.perf_counter()
            part = member.encode_paths([str(p) for p in ref_paths])
            ref_encode_s += time.perf_counter() - t0
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, part)
            print(f"[{args.tag}] encoded {len(ref_paths)} refs in {time.perf_counter() - t0:.1f}s"
                  + (f" ({name})" if ensemble else ""))
        ref_parts.append(np.ascontiguousarray(part, dtype=np.float32))
    ref_desc = combine_blocks(ref_parts)
    # The cache holds raw descriptors; centering is applied here, per run. An
    # ensemble is centred member by member so every member keeps an equal vote.
    centering = DomainCentering(args.center, args.center_warmup, blocks_of(encoder))
    ref_desc = centering.fit_refs(ref_desc)

    # ---- simulated prior -----------------------------------------------
    # One draw per query, seeded, so runs with different encoders see the same
    # priors and their numbers are comparable.
    prior_lat = prior_lon = None
    if args.prior_sigma:
        rng = np.random.default_rng(args.prior_seed)
        offsets = rng.normal(0.0, args.prior_sigma, size=(len(query_paths), 2))
        prior_lat = np.empty(len(query_paths))
        prior_lon = np.empty(len(query_paths))
        for i in range(len(query_paths)):
            prior_lat[i], prior_lon[i] = _from_local_xy(
                offsets[i, 0], offsets[i, 1], q_lat[i], q_lon[i]
            )
        print(f"[{args.tag}] search window: prior sigma {args.prior_sigma:.0f} m, "
              f"radius {args.prior_k:g} sigma = {args.prior_k * args.prior_sigma:.0f} m")

    # ---- per-frame plan: heading rotation + AGL scale (avl.pipeline) -----------
    agl = np.full(len(query_paths), np.nan)
    tile_m: float | None = None
    scale_info: dict = {"enabled": False}
    if args.scale_from_agl:
        import re

        from avl.terrain import TerrainModel

        if args.camera_k is None:
            raise SystemExit("--scale-from-agl needs --camera-k (scripts/visloc_scale.py calibrate)")
        tile_m = args.tile_m
        if tile_m is None:
            match = re.search(r"_t(\d+)", str(args.refs.parent.name))
            if not match:
                raise SystemExit("--scale-from-agl: pass --tile-m (no '_t<metres>' in the refs path)")
            tile_m = float(match.group(1))
        # In flight the terrain is looked up where the navigation prior says the
        # vehicle is; without a prior, at the logged position.
        look_lat = prior_lat if prior_lat is not None else q_lat
        look_lon = prior_lon if prior_lon is not None else q_lon
        terrain = TerrainModel.for_bounds(float(np.min(look_lat)), float(np.max(look_lat)),
                                          float(np.min(look_lon)), float(np.max(look_lon)), args.dem_dir)
        agl = queries[args.height_col].to_numpy(dtype=np.float64) - terrain.elevation(look_lat, look_lon)
        n_variants = args.rotations

    plans = []
    for i, path in enumerate(query_paths):
        width = height = 0
        if args.scale_from_agl:
            with Image.open(path) as handle:
                width, height = handle.size
        plans.append(plan_query(
            width, height,
            rotate_deg=float(q_yaw[i]) if np.isfinite(q_yaw[i]) else None,
            heading_mode=heading_mode,
            agl_m=float(agl[i]) if args.scale_from_agl else None,
            camera_k=args.camera_k if args.scale_from_agl else None,
            tile_m=tile_m,
            agl_gate=args.agl_gate,
            min_agl_m=args.min_agl,
            scales=tuple(args.query_scales),
        ))

    if args.scale_from_agl:
        fractions = np.asarray([p.scales[0] for p in plans])
        scale_info = {
            "enabled": True, "camera_k": args.camera_k, "tile_m": tile_m, "gate": args.agl_gate,
            "agl_median_m": float(np.nanmedian(agl)),
            "crop_fraction_median": float(np.median(fractions)),
            "uncropped_share": float(np.mean(fractions >= 0.9999)),
        }
        print(f"[{args.tag}] AGL scale: tile {tile_m:.0f} m, k {args.camera_k:.3f}, AGL median "
              f"{scale_info['agl_median_m']:.0f} m, crop fraction median "
              f"{scale_info['crop_fraction_median']:.2f}, {100 * scale_info['uncropped_share']:.0f}% uncropped")

    # ---- query side -----------------------------------------------------
    # One query cache per member, like the reference side.
    cached_q: list[np.ndarray | None] = []
    for member, ref_part in zip(members, ref_parts):
        name = member.config.model
        cache = member_path(args.query_cache, args.model, name)
        part = None
        if cache is not None and cache.exists():
            part = np.load(cache).astype(np.float32)
            if part.shape[:2] != (len(query_paths), n_variants) or part.shape[2] != ref_part.shape[1]:
                print(f"[{args.tag}] query cache {part.shape} does not fit this run; re-encoding"
                      + (f" ({name})" if ensemble else ""))
                part = None
            else:
                print(f"[{args.tag}] loaded cached query descriptors {part.shape}"
                      + (f" ({name})" if ensemble else ""))
        cached_q.append(part)
    all_cached = all(part is not None for part in cached_q)
    encoded_q: list[list[np.ndarray]] = [[] for _ in members]

    records = []
    t0 = time.perf_counter()
    print(f"[{args.tag}]   0/{len(query_paths)} queries", flush=True)
    for i, path in enumerate(query_paths):
        image = None
        plan = plans[i]
        if not all_cached or args.rerank:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            variants = query_variants(image, args.rotations, args.query_crop,
                                      plan.scales, plan.rotate_deg)
            image = variants[0]  # the upright, full-scale view, for re-ranking
        parts = []
        for m, member in enumerate(members):
            if cached_q[m] is not None:
                parts.append(cached_q[m][i])
            else:
                part = member.encode_images(variants, args.batch_size)
                encoded_q[m].append(part)
                parts.append(part)
        desc = centering.query(combine_blocks(parts))

        scores_all = desc @ ref_desc.T  # (R, N) cosine similarity
        best_per_ref = scores_all.max(axis=0)  # best orientation per reference tile

        # Re-ranking only re-orders what retrieval hands it, so the short-list has
        # to be deeper than the answer when the stage is on.
        candidate_k = max(args.top_k, args.rerank_candidates) if args.rerank else args.top_k

        window: np.ndarray | None = None
        if prior_lat is not None:
            window = search_window(
                ref_lat, ref_lon, prior_lat[i], prior_lon[i],
                args.prior_k * args.prior_sigma, min_keep=candidate_k,
            )
            best_per_ref = np.where(window, best_per_ref, -np.inf)

        order = np.argsort(-best_per_ref)
        seen: set[str] = set()
        top_idx: list[int] = []
        for idx in order:
            if not np.isfinite(best_per_ref[idx]):
                break
            rid = ref_ids[idx]
            if rid in seen:
                continue
            seen.add(rid)
            top_idx.append(int(idx))
            if len(top_idx) >= candidate_k:
                break

        rerank_ms = 0.0
        verified = 0
        retrieval_rank = {j: r + 1 for r, j in enumerate(top_idx)}
        fusion_weights: np.ndarray | None = None
        if args.rerank:
            t_rr = time.perf_counter()
            outcome = reranker.rerank(
                image,  # the query exactly as it was encoded
                [str(ref_paths[j]) for j in top_idx],
                [float(best_per_ref[j]) for j in top_idx],
            )
            rerank_ms = (time.perf_counter() - t_rr) * 1000.0
            verified = outcome.verified
            reordered = [top_idx[p] for p in outcome.order]
            weights_by_ref = {
                top_idx[s.candidate]: s.fusion_weight for s in outcome.stats
            }
            top_idx = reordered[: args.top_k]
            fusion_weights = np.asarray(
                [weights_by_ref[j] for j in top_idx], dtype=np.float64
            )

        top_idx = top_idx[: args.top_k]
        top_idx_arr = np.asarray(top_idx)
        top_scores = best_per_ref[top_idx_arr]

        errs = np.array(
            [haversine_m(q_lat[i], q_lon[i], ref_lat[j], ref_lon[j]) for j in top_idx_arr]
        )
        pose = weighted_geo_fusion(
            ref_lat[top_idx_arr],
            ref_lon[top_idx_arr],
            top_scores if fusion_weights is None else fusion_weights,
            method=args.fusion,
        )
        fused_err = haversine_m(q_lat[i], q_lon[i], pose.latitude, pose.longitude)
        spread = float(
            np.mean([haversine_m(pose.latitude, pose.longitude, ref_lat[j], ref_lon[j]) for j in top_idx_arr])
        )

        prior_cols = {}
        if window is not None:
            in_win = np.array(
                [haversine_m(q_lat[i], q_lon[i], ref_lat[j], ref_lon[j]) for j in np.flatnonzero(window)]
            )
            prior_cols = {
                "prior_lat": prior_lat[i],
                "prior_lon": prior_lon[i],
                "prior_error_m": haversine_m(q_lat[i], q_lon[i], prior_lat[i], prior_lon[i]),
                "window_tiles": int(window.sum()),
                "window_oracle_m": float(in_win.min()),
                # expected error of picking a tile uniformly at random from the window
                **{f"window_random_within_{t}": float((in_win <= t).mean()) for t in THRESHOLDS_M},
                "window_random_mean_m": float(in_win.mean()),
            }

        records.append(
            {
                "query_id": str(queries.iloc[i].get("image_id", i)),
                "gt_lat": q_lat[i],
                "gt_lon": q_lon[i],
                "top1_id": ref_ids[top_idx_arr[0]],
                "top1_score": float(top_scores[0]),
                "top2_score": float(top_scores[1]) if len(top_scores) > 1 else float("nan"),
                "score_margin": float(top_scores[0] - top_scores[1]) if len(top_scores) > 1 else float("nan"),
                "topk_ids": "|".join(ref_ids[j] for j in top_idx_arr),
                "topk_errors_m": "|".join(f"{e:.1f}" for e in errs),
                "topk_scores": "|".join(f"{s:.4f}" for s in top_scores),
                "top1_error_m": float(errs[0]),
                "best_topk_error_m": float(errs.min()),
                "fused_error_m": fused_err,
                "fused_lat": pose.latitude,
                "fused_lon": pose.longitude,
                "spread_m": spread,
                "confidence": float(top_scores[0] * np.exp(-spread / 500.0)),
                "rerank_verified": verified,
                "rerank_ms": rerank_ms,
                "top1_retrieval_rank": retrieval_rank.get(int(top_idx_arr[0])),
                # how the frame was prepared (avl.pipeline.QueryPlan)
                "rotate_deg": plan.rotate_deg if plan.rotate_deg is not None else float("nan"),
                "crop_fraction": plan.crop_fraction,
                "agl_m": plan.agl_m if plan.agl_m is not None else float("nan"),
                **prior_cols,
            }
        )
        if (i + 1) % 10 == 0 or (i + 1) == len(query_paths):
            print(f"[{args.tag}]   {i + 1}/{len(query_paths)} queries", flush=True)
    query_s = time.perf_counter() - t0
    if args.query_cache:
        for member, part, fresh in zip(members, cached_q, encoded_q):
            if part is None and fresh:
                cache = member_path(args.query_cache, args.model, member.config.model)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, np.stack(fresh))

    df = pd.DataFrame(records)
    per_query_csv = args.out_dir / f"{args.tag}_per_query.csv"
    df.to_csv(per_query_csv, index=False)

    def stats(col: str) -> dict:
        v = df[col].to_numpy(dtype=np.float64)
        return {
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
            "min": float(v.min()),
            "max": float(v.max()),
        }

    # Chance baseline: expected error of a uniformly random reference tile, and the
    # error of the single best tile in the map. Without these the absolute metres are
    # uninterpretable, because a small map makes even random guessing look accurate.
    lat0 = np.radians(float(np.mean(q_lat)))
    m_lat = 111_132.92 - 559.82 * np.cos(2 * lat0) + 1.175 * np.cos(4 * lat0)
    m_lon = 111_412.84 * np.cos(lat0) - 93.5 * np.cos(3 * lat0)
    dy = (ref_lat[None, :] - q_lat[:, None]) * m_lat
    dx = (ref_lon[None, :] - q_lon[:, None]) * m_lon
    dist = np.sqrt(dx**2 + dy**2)
    chance = {
        "random_tile_mean_m": float(dist.mean()),
        "random_tile_median_m": float(np.median(dist)),
        "oracle_best_tile_median_m": float(np.median(dist.min(axis=1))),
        "random_tile_within_m": {
            str(t): float((dist <= t).mean()) for t in THRESHOLDS_M
        },
    }

    # inter-reference spacing puts the error numbers in context
    summary = {
        "tag": args.tag,
        "model": args.model,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "refs_csv": str(args.refs),
        "queries_csv": str(args.queries),
        "n_refs": int(len(ref_paths)),
        "n_queries": int(len(df)),
        "rotations": args.rotations,
        "query_stride": args.query_stride,
        "max_queries": args.max_queries,
        "north_align": bool(args.north_align),
        "north_align_mode": heading_mode if args.north_align else None,
        "yaw_col": args.yaw_col if args.north_align else None,
        "yaw_sign": args.yaw_sign if args.north_align else None,
        "query_scales": list(args.query_scales),
        "scale_from_agl": scale_info,
        "query_crop": args.query_crop,
        "fusion": args.fusion,
        "members": [m.config.model for m in members],
        "descriptor_dim": int(ref_desc.shape[1]),
        "rerank": (
            {
                "enabled": True,
                "backend": args.rerank_backend,
                "candidates": args.rerank_candidates,
                "min_inliers": args.rerank_min_inliers,
                "blend": args.rerank_blend,
                "ms_per_query": float(df["rerank_ms"].mean()),
                "mean_verified": float(df["rerank_verified"].mean()),
                "queries_with_no_verification": int((df["rerank_verified"] == 0).sum()),
                "top1_promoted": int((df["top1_retrieval_rank"] > 1).sum()),
            }
            if args.rerank
            else {"enabled": False}
        ),
        "chance_baseline": chance,
        "prior": (
            {
                "enabled": True,
                "sigma_m": args.prior_sigma,
                "k": args.prior_k,
                "radius_m": args.prior_k * args.prior_sigma,
                "seed": args.prior_seed,
                "window_tiles_mean": float(df["window_tiles"].mean()),
                # the prior on its own: the bar retrieval must clear to be worth running
                "prior_error_m": stats("prior_error_m"),
                "prior_within_m": {
                    str(t): float((df["prior_error_m"] <= t).mean()) for t in THRESHOLDS_M
                },
                # a random tile from inside the window: the bar for "retrieval adds
                # information beyond the window itself"
                "window_random_within_m": {
                    str(t): float(df[f"window_random_within_{t}"].mean()) for t in THRESHOLDS_M
                },
                "window_oracle_within_m": {
                    str(t): float((df["window_oracle_m"] <= t).mean()) for t in THRESHOLDS_M
                },
            }
            if prior_lat is not None
            else {"enabled": False}
        ),
        "center": (
            {"mode": args.center, "warmup": args.center_warmup}
            if args.center == "map+flight"
            else {"mode": args.center}
        ),
        "top_k": args.top_k,
        "reference_encode_s": ref_encode_s,
        "query_total_s": query_s,
        "query_ms_per_image": 1000.0 * query_s / max(1, len(df)),
        # a cached run skips encoding, so its per-image time is not a latency figure
        "query_cache_hit": all_cached,
        "top1_error_m": stats("top1_error_m"),
        "fused_error_m": stats("fused_error_m"),
        "best_topk_error_m": stats("best_topk_error_m"),
        "top1_score": stats("top1_score"),
        "score_margin": stats("score_margin"),
        "recall_top1_within_m": {
            str(t): float((df["top1_error_m"] <= t).mean()) for t in THRESHOLDS_M
        },
        "recall_topk_within_m": {
            str(t): float((df["best_topk_error_m"] <= t).mean()) for t in THRESHOLDS_M
        },
        "fused_within_m": {
            str(t): float((df["fused_error_m"] <= t).mean()) for t in THRESHOLDS_M
        },
    }
    summary_path = args.out_dir / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[{args.tag}] {len(df)} queries vs {len(ref_paths)} reference tiles")
    print(f"  top-1 error   median {summary['top1_error_m']['median']:8.1f} m   "
          f"p95 {summary['top1_error_m']['p95']:8.1f} m")
    print(f"  fused error   median {summary['fused_error_m']['median']:8.1f} m   "
          f"p95 {summary['fused_error_m']['p95']:8.1f} m")
    print("  top-1 within: " + "  ".join(
        f"{t}m {100 * summary['recall_top1_within_m'][str(t)]:5.1f}%" for t in THRESHOLDS_M))
    print("  top-5 within: " + "  ".join(
        f"{t}m {100 * summary['recall_topk_within_m'][str(t)]:5.1f}%" for t in THRESHOLDS_M))
    print(f"  fused within ({args.fusion}): " + "  ".join(
        f"{t}m {100 * summary['fused_within_m'][str(t)]:5.1f}%" for t in THRESHOLDS_M))
    print(f"  chance        random tile median {chance['random_tile_median_m']:8.1f} m   "
          f"oracle best tile median {chance['oracle_best_tile_median_m']:6.1f} m")
    print("  chance within:" + "  ".join(
        f"{t}m {100 * chance['random_tile_within_m'][str(t)]:5.1f}%" for t in THRESHOLDS_M))
    if prior_lat is not None:
        pr = summary["prior"]
        print(f"  window        {pr['window_tiles_mean']:.1f} tiles on average  "
              f"(sigma {args.prior_sigma:.0f} m, radius {pr['radius_m']:.0f} m)")
        print("  prior alone:  " + "  ".join(
            f"{t}m {100 * pr['prior_within_m'][str(t)]:5.1f}%" for t in THRESHOLDS_M)
            + f"   median {pr['prior_error_m']['median']:.0f} m")
        print("  random in win:" + "  ".join(
            f"{t}m {100 * pr['window_random_within_m'][str(t)]:5.1f}%" for t in THRESHOLDS_M))
        print("  oracle in win:" + "  ".join(
            f"{t}m {100 * pr['window_oracle_within_m'][str(t)]:5.1f}%" for t in THRESHOLDS_M))
    if args.rerank:
        rr = summary["rerank"]
        print(f"  rerank        {rr['backend']}  {rr['ms_per_query']:.0f} ms/query  "
              f"{rr['mean_verified']:.1f}/{rr['candidates']} verified on average  "
              f"{rr['queries_with_no_verification']}/{len(df)} queries verified nothing  "
              f"top-1 promoted on {rr['top1_promoted']}/{len(df)}")
    print(f"  wrote {summary_path} and {per_query_csv}")


if __name__ == "__main__":
    main()
