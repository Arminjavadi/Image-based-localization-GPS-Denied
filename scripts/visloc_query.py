"""Localize one image against a prepared reference database.

The single-image counterpart to scripts/visloc_eval.py: same encoder, same
orientation search, same top-5 geo fusion, but one query and a JSON answer. The
benchmark console drives this out of process so the GUI never imports Torch.

It takes the same recipe as the benchmark (avl.pipeline): an ensemble ``a+b+c``,
north-up rotation from ``--heading-deg``, the altitude crop from ``--altitude-asl``
(or ``--agl``), map centering, and a search window around ``--prior-lat/--prior-lon``.
Flight-mean centering needs a sequence, so a single image is centred on the map mean.

Example
-------
python scripts/visloc_query.py --refs data/visloc_avl/r10_t200/references.csv \
    --image data/UAVVisLoc/10/drone/10_0013.JPG --model denseuav-vit

python scripts/visloc_query.py --refs data/visloc_avl/r05_t250/references.csv \
    --image data/UAVVisLoc/05/drone/05_0001.JPG --model megaloc --rotations 1 \
    --heading-deg 91.1 --heading-mode auto --altitude-asl 2315 --camera-k 0.971 \
    --center map --gt-lat 24.66589 --gt-lon 102.34125
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from avl.centering import CENTER_MODES, DomainCentering
from avl.ensemble import blocks_of, validate_model_spec
from avl.geo import DEFAULT_FUSION_METHOD, FUSION_METHODS, search_window
from avl.pipeline import HEADING_MODES, plan_query, rotation_for_heading, tile_m_of
from avl.rerank import DEFAULT_BACKEND, RERANK_BACKENDS, RerankConfig, Reranker
from avl.retrieval import (
    build_encoder,
    encode_reference_map,
    load_reference_map,
    localize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", default="denseuav-vit",
                        help="Encoder name, or an ensemble 'a+b+c'.")
    parser.add_argument("--rotations", type=int, default=4, choices=[1, 4])
    parser.add_argument("--query-crop", default="square", choices=["none", "square"])
    parser.add_argument(
        "--fusion",
        default=DEFAULT_FUSION_METHOD,
        choices=list(FUSION_METHODS),
        help="How the top-K matches are collapsed into one pose.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gt-lat", type=float, default=None)
    parser.add_argument("--gt-lon", type=float, default=None)
    parser.add_argument("--ref-cache", type=Path, default=None, help="Reference descriptor .npy")
    parser.add_argument("--vocab", type=Path, default=None, help="AnyLoc vocabulary .npz")
    parser.add_argument("--out", type=Path, default=None, help="Write the JSON result here too")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Verify the retrieval short-list with local features + RANSAC.",
    )
    parser.add_argument("--rerank-backend", choices=list(RERANK_BACKENDS), default=DEFAULT_BACKEND)
    parser.add_argument("--rerank-candidates", type=int, default=10)
    parser.add_argument("--rerank-min-inliers", type=int, default=12)
    parser.add_argument("--rerank-blend", type=float, default=0.5)
    # -- recipe steps (avl.pipeline) ------------------------------------------
    parser.add_argument("--heading-deg", type=float, default=None,
                        help="Compass direction of the image's top edge, degrees clockwise from north.")
    parser.add_argument("--heading-mode", default="auto", choices=list(HEADING_MODES),
                        help="How --heading-deg turns the frame north-up (see avl.pipeline.plan_query).")
    parser.add_argument("--altitude-asl", type=float, default=None,
                        help="Altitude above sea level; AGL = this - terrain under the prior "
                             "(or under --gt-lat/--gt-lon when there is no prior).")
    parser.add_argument("--agl", type=float, default=None, help="Height above ground, if known directly.")
    parser.add_argument("--camera-k", type=float, default=None,
                        help="Full-frame ground width / AGL; enables the altitude crop.")
    parser.add_argument("--tile-m", type=float, default=None,
                        help="Tile ground size; default parsed from '_t<metres>' in --refs.")
    parser.add_argument("--agl-gate", type=float, default=1.25)
    parser.add_argument("--min-agl", type=float, default=100.0)
    parser.add_argument("--dem-dir", type=Path, default=Path("data/dem/copernicus_glo30"))
    parser.add_argument("--center", default="off", choices=list(CENTER_MODES),
                        help="Domain centering; map+flight falls back to the map mean for one image.")
    parser.add_argument("--prior-lat", type=float, default=None)
    parser.add_argument("--prior-lon", type=float, default=None)
    parser.add_argument("--prior-sigma", type=float, default=None,
                        help="1-sigma of the prior (m); search only tiles within --prior-k sigma.")
    parser.add_argument("--prior-k", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Query image not found: {args.image}")

    try:
        validate_model_spec(args.model)
    except ValueError as error:
        raise SystemExit(f"--model: {error}")
    references = load_reference_map(args.refs)
    print(f"[query] {len(references)} reference tiles from {args.refs}", flush=True)

    started = time.perf_counter()
    encoder = build_encoder(args.model, references, vocab_path=args.vocab,
                            batch_size=args.batch_size, rotations=args.rotations)
    print(f"[query] {args.model} ready on {encoder.device} "
          f"in {time.perf_counter() - started:.1f}s", flush=True)

    started = time.perf_counter()
    ref_desc, cached = encode_reference_map(encoder, references, args.ref_cache, args.batch_size)
    print(f"[query] reference descriptors {ref_desc.shape} "
          f"{'reused from cache' if cached else f'encoded in {time.perf_counter() - started:.1f}s'}",
          flush=True)

    ground_truth = (
        (args.gt_lat, args.gt_lon) if args.gt_lat is not None and args.gt_lon is not None else None
    )
    prior = (
        (args.prior_lat, args.prior_lon)
        if args.prior_lat is not None and args.prior_lon is not None
        else None
    )

    # A single frame has no flight to average over: centre it on the map mean.
    center_mode = "map" if args.center == "map+flight" else args.center
    centering = DomainCentering(center_mode, blocks=blocks_of(encoder))
    ref_desc = centering.fit_refs(ref_desc)

    agl = args.agl
    look = prior or ground_truth
    if agl is None and args.altitude_asl is not None and args.camera_k is not None and look:
        from avl.terrain import TerrainModel

        terrain = TerrainModel.for_bounds(look[0], look[0], look[1], look[1], args.dem_dir)
        agl = float(args.altitude_asl - terrain.elevation(look[0], look[1])[0])
    tile_m = args.tile_m or tile_m_of(args.refs)
    with Image.open(args.image) as handle:
        width, height = handle.size
    plan = plan_query(
        width, height,
        rotate_deg=rotation_for_heading(args.heading_deg),
        heading_mode=args.heading_mode,
        agl_m=agl,
        camera_k=args.camera_k,
        tile_m=tile_m if args.camera_k is not None else None,
        agl_gate=args.agl_gate,
        min_agl_m=args.min_agl,
    )
    window = None
    if prior is not None and args.prior_sigma:
        window = search_window(
            references.latitude, references.longitude, prior[0], prior[1],
            args.prior_k * args.prior_sigma, min_keep=args.top_k,
        )
    print(f"[query] plan: rotate {plan.rotate_deg if plan.rotate_deg is not None else '-'}°  "
          f"crop {plan.crop_fraction:.2f}  AGL {f'{agl:.0f} m' if agl is not None else '-'}  "
          f"center {center_mode}  window {int(window.sum()) if window is not None else 'all'} tiles",
          flush=True)

    result = localize(
        encoder,
        args.image,
        references,
        ref_desc,
        rotations=args.rotations,
        query_crop=args.query_crop,
        top_k=args.top_k,
        ground_truth=ground_truth,
        batch_size=args.batch_size,
        fusion_method=args.fusion,
        reranker=Reranker(
            RerankConfig(
                enabled=args.rerank,
                backend=args.rerank_backend,
                candidates=args.rerank_candidates,
                min_inliers=args.rerank_min_inliers,
                blend=args.rerank_blend,
                device=str(encoder.device),
            )
        ),
        window=window,
        scales=plan.scales,
        heading_deg=plan.rotate_deg,
        centering=centering,
    )
    result.plan = {
        **plan.as_dict(),
        "heading_deg": args.heading_deg,
        "center": center_mode,
        "window_tiles": int(window.sum()) if window is not None else None,
        "members": validate_model_spec(args.model),
    }

    payload = result.as_dict()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))

    best = result.matches[0]
    print(f"[query] fusion={args.fusion}", flush=True)
    if result.rerank:
        print(f"[query] rerank={result.rerank['backend']} "
              f"{result.rerank['verified']}/{result.rerank['candidates']} verified "
              f"in {result.rerank['elapsed_ms']:.0f} ms"
              + (f"   ({result.rerank['error']})" if result.rerank["error"] else ""),
              flush=True)
    print(f"[query] fused {result.fused.latitude:.6f}, {result.fused.longitude:.6f}   "
          f"top-1 {best.image_id} {best.latitude:.6f}, {best.longitude:.6f} score {best.score:.3f}"
          + (f"   error {result.fused_error_m:.0f} m" if result.fused_error_m is not None else ""),
          flush=True)
    print("RESULT_JSON " + json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
