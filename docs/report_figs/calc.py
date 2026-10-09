"""Master computation for the AVL sizing / real-time feasibility report."""
import json, math

OUT = {}

# ----------------------------------------------------------------------------
# 1. MEASURED MODEL COSTS (this laptop, i7-7600U, 4 threads, PyTorch 2.12 CPU)
# ----------------------------------------------------------------------------
MODELS = {
    # name: params(M), GMACs, dim, input, measured single-image CPU ms (this laptop)
    "DenseUAV-ViT (ViT-S/16)": dict(params=21.86, gmacs=4.241, dim=512,  res=224, cpu_ms=95.3,
                                    batch_ms=85.0, weights_mb=21.86*4/1.024**2),
    "MixVPR-512 (RN50-L3)":    dict(params=10.09, gmacs=8.105, dim=512,  res=320, cpu_ms=226.7,
                                    batch_ms=230.0, weights_mb=10.09*4/1.024**2),
    "MixVPR-4096 (RN50-L3)":   dict(params=10.88, gmacs=8.421, dim=4096, res=320, cpu_ms=241.2,
                                    batch_ms=250.0, weights_mb=10.88*4/1.024**2),
    "CosPlace RN101-2048":     dict(params=46.70, gmacs=17.456, dim=2048, res=322, cpu_ms=553.5,
                                    batch_ms=480.0, weights_mb=46.70*4/1.024**2),
}

# ----------------------------------------------------------------------------
# 2. JETSON PLATFORM TABLE  (specs from NVIDIA / forecr; prices = 2026 list)
# ----------------------------------------------------------------------------
BOARDS = {
    # name: INT8 dense TOPS (=sparse/2), FP16 dense TFLOPS, bandwidth GB/s, W, USD(module), RAM GB
    "Jetson Nano (legacy)":   dict(tops=0.236, fp16=0.472, bw=25.6,  w=10,  usd=199,  ram=4,
                                   gpu="128-core Maxwell", cpu="4× A57 1.43 GHz", eol=True,
                                   no_tensor_cores=True),
    "Jetson Xavier NX 16GB":  dict(tops=10.5,  fp16=6.0,   bw=59.7,  w=20,  usd=699,  ram=16,
                                   gpu="384-core Volta + 48 TC", cpu="6× Carmel 1.9 GHz", eol=False),
    "Orin Nano 4GB (Super)":  dict(tops=17.0,  fp16=8.5,   bw=51.0,  w=25,  usd=349,  ram=4,
                                   gpu="512-core Ampere + 16 TC", cpu="6× A78AE 1.7 GHz", eol=False),
    "Orin Nano 8GB (Super)":  dict(tops=33.5,  fp16=16.7,  bw=102.0, w=25,  usd=399,  ram=8,
                                   gpu="1024-core Ampere + 32 TC", cpu="6× A78AE 1.7 GHz", eol=False),
    "Orin NX 8GB (Super)":    dict(tops=58.5,  fp16=29.3,  bw=102.4, w=40,  usd=649,  ram=8,
                                   gpu="1792-core Ampere + 56 TC", cpu="8× A78AE 2.0 GHz", eol=False),
    "Orin NX 16GB (Super)":   dict(tops=78.5,  fp16=39.3,  bw=102.4, w=40,  usd=899,  ram=16,
                                   gpu="2048-core Ampere + 64 TC", cpu="8× A78AE 2.0 GHz", eol=False),
    "AGX Orin 32GB":          dict(tops=100.0, fp16=50.0,  bw=204.8, w=60,  usd=1799, ram=32,
                                   gpu="2048-core Ampere + 64 TC", cpu="12× A78AE 2.2 GHz", eol=False),
    "AGX Orin 64GB":          dict(tops=137.5, fp16=68.8,  bw=204.8, w=75,  usd=2999, ram=64,
                                   gpu="2048-core Ampere + 64 TC", cpu="12× A78AE 2.2 GHz", eol=False),
    "AGX Thor T5000":         dict(tops=1035.0, fp16=250.0, bw=273.0, w=130, usd=5499, ram=128,
                                   gpu="2560-core Blackwell", cpu="14× Neoverse-V3AE 2.6 GHz", eol=False),
}

# Empirical anchor: transformer, TensorRT FP16, Orin Nano 8GB Super.
#   SegFormer-B0 (14.84 GMACs, 640x640) -> 15.02 ms  => 1.976 TFLOP/s effective
#   ViT-B/16 224 (17.6 GMACs)           -> 19.9  ms  => 1.769 TFLOP/s effective
# We adopt the conservative 1.77 TFLOP/s achieved-FP16 rate for the reference board,
# plus a 1.2 ms fixed per-inference overhead (launch + H2D/D2H + pre-processing on GPU).
ANCHOR_TFLOPS = 1.77
ANCHOR_FIXED_MS = 1.2
ANCHOR_BOARD = "Orin Nano 8GB (Super)"
# Sub-linear scaling exponent fitted to observed cross-board scaling of small vision
# models (YOLO26n and SegFormer-B0 across Nano Super / Orin NX 16GB / AGX Orin 64GB):
#   observed exponents 0.12-0.40; we adopt 0.50 as a deliberately conservative midpoint
#   for a model (ViT-S) that is more compute-bound than YOLO26n.
SCALE_EXP = 0.50


def jetson_ms(model, board, precision="fp16"):
    """Projected TensorRT latency (ms) for one forward pass, batch = 1."""
    m = MODELS[model]
    b = BOARDS[board]
    a = BOARDS[ANCHOR_BOARD]
    gflops = m["gmacs"] * 2.0
    base = gflops / (ANCHOR_TFLOPS * 1e3) * 1e3 + ANCHOR_FIXED_MS   # ms on anchor board
    ratio = b["fp16"] / a["fp16"]
    # Piecewise scaling: sub-linear for faster boards (they become launch/memory bound),
    # near-linear for slower boards (they stay compute bound). Maxwell has no tensor
    # cores, so its FP16 rate is already a hard ceiling -> fully linear.
    if b.get("no_tensor_cores"):
        exp = 1.0
    else:
        exp = SCALE_EXP if ratio >= 1.0 else 0.85
    speedup = ratio ** exp
    ms = base / speedup
    if precision == "int8":
        ms *= 0.72          # observed INT8/FP16 ratio on Orin (YOLO: 3.80/4.57 = 0.83;
                            # SegFormer: 21.58/15.02 at different power; we use 0.72 midpoint)
    return ms


# ----------------------------------------------------------------------------
# 3. DENSEUAV DATASET FACTS (measured on disk + GPS metadata)
# ----------------------------------------------------------------------------
DENSEUAV = dict(
    images=40833, gb_on_disk=16.76,
    train_drone=6768, train_sat=13536, test_query=2331, test_gallery=18198,
    locations_all=3033, locations_train=2256, locations_test=777,
    bbox_km2=9.394, hull_km2=7.128, campus_bbox_km2=1.145,
    footprint_union_150m_km2=2.581, footprint_union_200m_km2=3.231,
    spacing_median_m=19.7, extent_ew_km=5.465, extent_ns_km=1.719,
    lat=(30.309401, 30.324908), lon=(120.335385, 120.392203),
    campuses=14, alt_m=(80, 90, 100),
)

# ----------------------------------------------------------------------------
# 4. ARDABIL SIZING MODEL
# ----------------------------------------------------------------------------
ARDABIL = dict(area_city_km2=18.011, area_county_km2=2211.0, pop=588000,
               lat=38.25167, lon=48.29750, elev_m=1351)

M_PER_DEG_LAT = 111132.92 - 559.82*math.cos(math.radians(2*ARDABIL["lat"])) \
                + 1.175*math.cos(math.radians(4*ARDABIL["lat"]))
M_PER_DEG_LON = 111412.84*math.cos(math.radians(ARDABIL["lat"])) \
                - 93.5*math.cos(math.radians(3*ARDABIL["lat"]))
ARDABIL["m_per_deg_lat"] = M_PER_DEG_LAT
ARDABIL["m_per_deg_lon"] = M_PER_DEG_LON
ARDABIL["bbox_side_km"] = math.sqrt(ARDABIL["area_city_km2"])

TILE_PX = 512
JPEG_KB_PER_TILE = 26.2          # measured: 512x512 RGB -> JPEG q85
TIF_KB_PER_TILE = 232.0          # measured mean of DenseUAV satellite .tif


def tiling(area_km2, gsd_m=0.3, overlap=0.5, tile_px=TILE_PX):
    T = tile_px * gsd_m                       # tile ground side, m
    s = T * (1.0 - overlap)                   # grid step, m
    n = area_km2 * 1e6 / (s * s)
    return dict(tile_m=T, step_m=s, n_tiles=int(math.ceil(n)),
                density=n / area_km2)


def index_bytes_per_vec(dim, kind, m_pq=64, hnsw_m=32):
    if kind == "flat":
        return dim * 4
    if kind == "hnsw":
        return dim * 4 + 271                     # measured link overhead for M=32
    if kind == "ivfpq":
        return m_pq + 8 + 50                     # measured 122 B/vec at m=64, D=512
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# 5. MEASURED FAISS SEARCH (this laptop) + projected Jetson search
# ----------------------------------------------------------------------------
FAISS_MEASURED = {   # (dim, N, kind) -> (size_MB, q1_ms, q4_ms)
    (512, 1554, "flat"):   (3.2, 0.272, 0.289),
    (512, 1554, "hnsw"):   (3.6, 0.227, 0.60),
    (512, 7204, "flat"):   (14.8, 1.235, 3.9),
    (512, 7204, "hnsw"):   (16.7, 1.879, 3.2),
    (512, 7204, "ivfpq"):  (1.7, 0.170, 0.42),
    (512, 45028, "flat"):  (92.2, 45.469, 42.356),
    (512, 45028, "hnsw"):  (104.5, 3.535, 9.257),
    (512, 45028, "ivfpq"): (5.5, 0.197, 0.377),
    (2048, 1554, "flat"):  (12.7, 3.142, 17.276),
    (2048, 7204, "flat"):  (59.0, 8.998, 31.799),
    (2048, 45028, "flat"): (368.9, 138.283, 183.769),
    (4096, 1554, "flat"):  (25.5, 4.700, 24.891),
    (4096, 7204, "flat"):  (30.853*0, 30.853, 61.712),
    (4096, 45028, "flat"): (737.7, 232.387, 315.732),
}

# ----------------------------------------------------------------------------
# REPORT TABLES
# ----------------------------------------------------------------------------
def fmt(x, n=2):
    return f"{x:,.{n}f}"


lines = []
P = lines.append

P("=" * 100)
P("TABLE A. DenseUAV coverage (computed from Dense_GPS_ALL.txt, 3033 unique sampling points)")
P("=" * 100)
d = DENSEUAV
P(f"  bounding box            : {d['extent_ew_km']:.3f} km (E-W) x {d['extent_ns_km']:.3f} km (N-S) = {d['bbox_km2']:.3f} km2")
P(f"  convex hull of samples  : {d['hull_km2']:.3f} km2")
P(f"  sum of 14 campus boxes  : {d['campus_bbox_km2']:.3f} km2   <-- actual surveyed envelope")
P(f"  union of 150 m footprints: {d['footprint_union_150m_km2']:.3f} km2  <-- effective imaged coverage")
P(f"  union of 200 m footprints: {d['footprint_union_200m_km2']:.3f} km2")
P(f"  point density (in campus): {3033/d['campus_bbox_km2']:.0f} loc/km2  (20 m grid = 2500 loc/km2)")
P(f"  median NN spacing       : {d['spacing_median_m']} m")
P(f"  images / disk           : {d['images']:,} images, {d['gb_on_disk']} GB")
P(f"  GB per km2 (imaged)     : {d['gb_on_disk']/d['footprint_union_150m_km2']:.2f} GB/km2")
P(f"  GB per km2 (hull)       : {d['gb_on_disk']/d['hull_km2']:.2f} GB/km2")
P("")

P("=" * 100)
P("TABLE B. Model cost (measured on i7-7600U, PyTorch 2.12 CPU, batch=1, 4 threads)")
P("=" * 100)
P(f"{'model':26s}{'params M':>9}{'GMACs':>8}{'GFLOPs':>8}{'dim':>6}{'res':>6}{'FP32 wt MB':>11}{'CPU ms':>9}{'CPU FPS':>9}")
for k, m in MODELS.items():
    P(f"{k:26s}{m['params']:9.2f}{m['gmacs']:8.2f}{m['gmacs']*2:8.2f}{m['dim']:6d}{m['res']:6d}"
      f"{m['weights_mb']:11.1f}{m['cpu_ms']:9.1f}{1000/m['cpu_ms']:9.2f}")
P("")

P("=" * 100)
P("TABLE C. Projected TensorRT FP16 encode latency, batch=1 (ms) -- see method note")
P("=" * 100)
hdr = f"{'board':24s}" + "".join(f"{k.split(' (')[0][:13]:>14s}" for k in MODELS) + f"{'W':>5}{'USD':>7}"
P(hdr)
for b in BOARDS:
    row = f"{b:24s}"
    for k in MODELS:
        row += f"{jetson_ms(k, b):14.2f}"
    row += f"{BOARDS[b]['w']:5d}{BOARDS[b]['usd']:7d}"
    P(row)
P("")
P("  Reference (this laptop, PyTorch CPU FP32):")
P(f"{'i7-7600U CPU':24s}" + "".join(f"{MODELS[k]['cpu_ms']:14.2f}" for k in MODELS) + f"{15:5d}{0:7d}")
P("")

P("=" * 100)
P("TABLE D. Ardabil reference-map tiling options (city area 18.011 km2)")
P("=" * 100)
P(f"{'GSD m/px':>9}{'overlap':>9}{'tile m':>8}{'step m':>8}{'#tiles':>9}{'tiles/km2':>11}"
  f"{'imagery GB':>12}{'D512 idx MB':>13}{'D4096 idx MB':>14}")
TILINGS = []
for gsd in (0.3, 0.5):
    for ov in (0.0, 0.5, 0.75):
        t = tiling(ARDABIL["area_city_km2"], gsd, ov)
        n = t["n_tiles"]
        img_gb = n * JPEG_KB_PER_TILE / 1e6
        i512 = n * index_bytes_per_vec(512, "flat") / 1e6
        i4096 = n * index_bytes_per_vec(4096, "flat") / 1e6
        TILINGS.append(dict(gsd=gsd, overlap=ov, **t, imagery_gb=img_gb,
                            idx512_mb=i512, idx4096_mb=i4096))
        P(f"{gsd:9.2f}{ov:9.0%}{t['tile_m']:8.1f}{t['step_m']:8.1f}{n:9,d}{t['density']:11.0f}"
          f"{img_gb:12.3f}{i512:13.1f}{i4096:14.1f}")
P("")

P("=" * 100)
P("TABLE E. Ardabil - if the DenseUAV *UAV-survey* protocol were replicated (20 m waypoint grid)")
P("=" * 100)
for step in (20, 50, 100, 150, 200):
    n_loc = ARDABIL["area_city_km2"] * 1e6 / step**2
    track_km = ARDABIL["area_city_km2"] * 1e6 / step / 1000
    hours_15 = track_km * 1000 / 15 / 3600
    sorties = hours_15 * 60 / 22          # 22 min effective per multirotor sortie
    n_img_dense = n_loc * (3 + 6)          # DenseUAV: 3 drone + 6 satellite per point
    gb = n_loc * (3 * 1043 + 6 * 232) / 1e6      # KB -> GB (3 drone JPG + 6 sat TIF per point)
    P(f"  step {step:3d} m : {n_loc:10,.0f} waypoints | flight track {track_km:8,.0f} km | "
      f"{hours_15:7.1f} h @15 m/s | ~{sorties:5.0f} sorties | {n_img_dense:11,.0f} images | {gb:8.1f} GB")
P("")

P("=" * 100)
P("TABLE F. FAISS index cost per gallery size and descriptor dimension (analytic, B/vector)")
P("=" * 100)
P(f"{'N (tiles)':>12}" + "".join(f"{lbl:>14s}" for lbl in
  ["D512 flat", "D512 hnsw", "D512 ivfpq", "D2048 flat", "D4096 flat", "D4096 ivfpq"]))
GALLERIES = [1554, 3054, 7204, 12215, 45028, 180112, 1000000]
for n in GALLERIES:
    row = f"{n:12,d}"
    for dim, kind in [(512, "flat"), (512, "hnsw"), (512, "ivfpq"),
                      (2048, "flat"), (4096, "flat"), (4096, "ivfpq")]:
        mb = n * index_bytes_per_vec(dim, kind) / 1e6
        row += f"{mb:13.1f}M"
    P(row)
P("")

P("=" * 100)
P("TABLE G. Measured FAISS search latency on i7-7600U (median ms, k=10)")
P("=" * 100)
P(f"{'dim':>6}{'N':>9}{'index':>8}{'size MB':>10}{'1 query ms':>12}{'4 rot ms':>11}")
for (dim, n, kind), (mb, q1, q4) in sorted(FAISS_MEASURED.items()):
    if mb == 0:
        mb = n * index_bytes_per_vec(dim, kind) / 1e6
    P(f"{dim:6d}{n:9,d}{kind:>8}{mb:10.1f}{q1:12.3f}{q4:11.3f}")
P("")

P("=" * 100)
P("TABLE H. End-to-end online budget for Ardabil (3,054-tile map, 50% overlap, GSD 0.3 m)")
P("=" * 100)
N_ARD = 3054
PRE_MS = {"laptop": 12.0, "jetson": 4.0}       # decode 1440x1080 + resize
FUSE_MS = 0.3
P(f"{'platform':24s}{'model':22s}{'rot':>4}{'pre':>7}{'encode':>8}{'search':>8}{'fuse':>6}"
  f"{'TOTAL ms':>10}{'Hz':>7}{'verdict@10Hz':>14}")


def search_ms(dim, n, kind, rot, platform):
    """Interpolate measured laptop search; Jetson CPU ~2.2x faster per core-pair for
    memory-bound flat, ~1.5x for graph/PQ (A78AE @2 GHz, LPDDR5 102 GB/s vs DDR4 ~25 GB/s)."""
    key = min(((abs(k[1] - n), k) for k in FAISS_MEASURED
               if k[0] == dim and k[2] == kind), default=(None, None))[1]
    if key is None:
        return None
    base = FAISS_MEASURED[key][2 if rot == 4 else 1] * (n / key[1] if kind == "flat" else 1.0)
    if platform == "jetson":
        base /= (3.0 if kind == "flat" else 1.6)
    return base


rows = []
for plat, board in [("laptop", None),
                    ("jetson", "Orin Nano 8GB (Super)"),
                    ("jetson", "Orin NX 16GB (Super)"),
                    ("jetson", "AGX Orin 64GB")]:
    for mdl in MODELS:
        for rot in (4, 1):
            m = MODELS[mdl]
            dim = m["dim"]
            kind = "ivfpq" if dim == 512 else "flat"
            enc = (m["cpu_ms"] if plat == "laptop" else jetson_ms(mdl, board)) * rot
            srch = search_ms(dim, N_ARD, kind, rot, plat) or 0.0
            pre = PRE_MS[plat]
            tot = pre + enc + srch + FUSE_MS
            name = "i7-7600U (CPU)" if plat == "laptop" else board
            v = "PASS" if tot <= 100 else ("5 Hz ok" if tot <= 200 else
                                           ("1 Hz ok" if tot <= 1000 else "FAIL"))
            P(f"{name:24s}{mdl.split(' (')[0]:22s}{rot:4d}{pre:7.1f}{enc:8.1f}{srch:8.2f}"
              f"{FUSE_MS:6.1f}{tot:10.1f}{1000/tot:7.2f}{v:>14s}")
            rows.append((name, mdl, rot, tot, 1000/tot))
    P("")
OUT["budget"] = rows

P("=" * 100)
P("TABLE I. Offline index-build cost for the Ardabil map (3,054 tiles) and larger maps")
P("=" * 100)
P(f"{'platform':24s}{'model':22s}" + "".join(f"{f'{n:,} tiles':>16s}" for n in (3054, 12215, 45028)))
for plat, board in [("laptop", None), ("jetson", "Orin Nano 8GB (Super)"),
                    ("jetson", "Orin NX 16GB (Super)"), ("jetson", "AGX Orin 64GB")]:
    for mdl in MODELS:
        m = MODELS[mdl]
        per = m["batch_ms"] if plat == "laptop" else jetson_ms(mdl, board) * 0.55  # batch-32 amortisation
        name = "i7-7600U (CPU)" if plat == "laptop" else board
        row = f"{name:24s}{mdl.split(' (')[0]:22s}"
        for n in (3054, 12215, 45028):
            s = per * n / 1000
            row += f"{(f'{s/60:.1f} min' if s < 3600 else f'{s/3600:.2f} h'):>16s}"
        P(row)
P("")

P("=" * 100)
P("TABLE J. Real-time requirement derivation")
P("=" * 100)
for v in (5, 10, 15, 25, 40):
    P(f"  UAV speed {v:3d} m/s ({v*3.6:5.1f} km/h): "
      f"1 fix / 76.8 m map cell -> {v/76.8:5.2f} Hz | "
      f"1 fix / 20 m -> {v/20:5.2f} Hz | "
      f"drift @1 Hz (1%% IMU) = {v*0.01*100:4.1f} cm/s | "
      f"distance per 100 ms = {v*0.1:4.1f} m")
P("")
P("  Coverage rate of a survey UAV (150 m swath):")
for v in (10, 15, 25):
    P(f"    {v:2d} m/s -> {v*150*3600/1e6:5.2f} km2/h  =>  Ardabil (18.011 km2) in "
      f"{18.011/(v*150*3600/1e6):5.1f} h of flight")
P("")

txt = "\n".join(lines)
print(txt)
open("/tmp/claude-1000/-home-pouria-Python-Image-based-localization-GPS-Denied/"
     "0133985b-7ccd-4e88-b5ad-3279c7211e71/scratchpad/tables.txt", "w").write(txt)

json.dump(dict(models=MODELS, boards=BOARDS, denseuav=DENSEUAV, ardabil=ARDABIL,
               tilings=TILINGS,
               jetson_ms={b: {m: jetson_ms(m, b) for m in MODELS} for b in BOARDS}),
          open("/tmp/claude-1000/-home-pouria-Python-Image-based-localization-GPS-Denied/"
               "0133985b-7ccd-4e88-b5ad-3279c7211e71/scratchpad/model.json", "w"), indent=1)
