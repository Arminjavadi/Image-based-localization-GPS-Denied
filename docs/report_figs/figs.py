import math, os, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})
C = dict(blue="#2b6cb0", teal="#2c7a7b", amber="#b7791f", red="#c53030",
         grey="#4a5568", green="#276749", purple="#6b46c1", light="#cbd5e0")

DEN = "/home/pouria/Python/Image-based-localization-GPS-Denied/data/DenseUAV/DenseUAV/"


def load_gps(fn):
    pts = {}
    for line in open(DEN + fn):
        p = line.split()
        if len(p) < 3:
            continue
        pts[(p[0].split('/')[0], p[0].split('/')[-2])] = (
            float(p[2].lstrip('N')), float(p[1].lstrip('E')), float(p[3]) if len(p) > 3 else 0)
    return list(pts.values())


# ============================================================ FIG 1 : coverage map
P = load_gps("Dense_GPS_ALL.txt")
Ptr = set((round(a, 7), round(b, 7)) for a, b, _ in load_gps("Dense_GPS_train.txt"))
lat0 = sum(p[0] for p in P) / len(P)
mlat = 111132.92 - 559.82*math.cos(math.radians(2*lat0)) + 1.175*math.cos(math.radians(4*lat0))
mlon = 111412.84*math.cos(math.radians(lat0)) - 93.5*math.cos(math.radians(3*lat0))
lo_min = min(p[1] for p in P); la_min = min(p[0] for p in P)
xs = np.array([(p[1]-lo_min)*mlon for p in P])/1000
ys = np.array([(p[0]-la_min)*mlat for p in P])/1000
is_tr = np.array([(round(p[0], 7), round(p[1], 7)) in Ptr for p in P])

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 5.4),
                             gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.42))
a1.scatter(xs[is_tr], ys[is_tr], s=2.0, c=C["blue"], label=f"train points (n={is_tr.sum()})", lw=0)
a1.scatter(xs[~is_tr], ys[~is_tr], s=2.0, c=C["amber"], label=f"test-only points (n={(~is_tr).sum()})", lw=0)
a1.add_patch(Rectangle((0, 0), xs.max(), ys.max(), fill=False, ec=C["red"], ls="--", lw=1.0))
a1.text(xs.max()*0.60, ys.max()*1.10,
        f"bounding box  {xs.max():.2f} × {ys.max():.2f} km = 9.39 km²",
        color=C["red"], fontsize=8)
a1.set_aspect("equal"); a1.set_xlabel("east (km)"); a1.set_ylabel("north (km)")
a1.set_title("(a)  DenseUAV sampling geometry — Xiasha university town, Hangzhou (30.32° N, 120.36° E)")
a1.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
a1.set_ylim(-0.35, 2.35)

# zoom on one campus
sel = (xs > 1.95) & (xs < 2.15) & (ys > 1.55) & (ys < 1.72)
a2b = a1.inset_axes([0.02, 0.60, 0.17, 0.36])
a2b.scatter(xs[sel]*1000, ys[sel]*1000, s=7, c=C["blue"], lw=0)
a2b.set_xticks([]); a2b.set_yticks([]); a2b.grid(False)
a2b.set_title("zoom: 20 m grid", fontsize=6.2, pad=1.5)
for s in a2b.spines.values():
    s.set_edgecolor(C["grey"])

# coverage-definition bars
labels = ["bounding\nbox", "convex hull\nof samples", "Σ 14 campus\nboxes", "union of\n150 m footprints",
          "union of\n200 m footprints"]
vals = [9.394, 7.128, 1.145, 2.581, 3.231]
cols = [C["light"], C["light"], C["blue"], C["teal"], C["teal"]]
b = a2.bar(labels, vals, color=cols, edgecolor=C["grey"], lw=0.6, width=0.6)
for r, v in zip(b, vals):
    a2.text(r.get_x()+r.get_width()/2, v+0.15, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
a2.set_ylabel("area (km²)"); a2.set_ylim(0, 11)
a2.set_title("(b)  Five defensible definitions of “dataset coverage area” for DenseUAV")
a2.tick_params(axis="x", labelsize=7.5)
a2.annotate("what the drone actually imaged", xy=(3.05, 3.05), xytext=(3.2, 6.4),
            fontsize=7.5, color=C["teal"], ha="center",
            arrowprops=dict(arrowstyle="->", color=C["teal"], lw=0.9))
fig.savefig(f"{FIG}/f1_coverage.png"); plt.close(fig)

# ============================================================ FIG 2 : dataset comparison
DS = [   # name, images(k), area km2, platform, colour
    ("CVUSA",           71.0,   None,  "ground-sat"),
    ("VIGOR",          144.0,   None,  "ground-sat"),
    ("University-1652", 50.2,   None,  "3-view"),
    ("SUES-200",         6.1,   None,  "drone-sat"),
    ("DenseUAV",        40.8,   2.58,  "drone-sat"),
    ("UAV-VisLoc",       6.7, 380.21,  "drone-sat"),
    ("VPAIR",            2.7,   None,  "aerial"),
    ("ALTO",            30.0,   None,  "aerial"),
    ("Ardabil (this work,\nproposed)", 3.1, 18.01, "sat-tiles"),
]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
names = [d[0] for d in DS]; imgs = [d[1] for d in DS]
cc = [C["purple"] if "Ardabil" in n else (C["blue"] if n == "DenseUAV" else C["light"]) for n in names]
a1.barh(range(len(DS)), imgs, color=cc, edgecolor=C["grey"], lw=0.5)
a1.set_yticks(range(len(DS))); a1.set_yticklabels(names, fontsize=7)
a1.invert_yaxis(); a1.set_xlabel("images (thousands)")
a1.set_title("(a)  Dataset size", fontsize=9)
for i, v in enumerate(imgs):
    a1.text(v+2, i, f"{v:.1f}k", va="center", fontsize=6.8)
a1.set_xlim(0, 175)

kn = [(d[0], d[2]) for d in DS if d[2]]
a2.barh(range(len(kn)), [k[1] for k in kn],
        color=[C["purple"] if "Ardabil" in k[0] else (C["blue"] if k[0] == "DenseUAV" else C["teal"]) for k in kn],
        edgecolor=C["grey"], lw=0.5)
a2.set_yticks(range(len(kn))); a2.set_yticklabels([k[0] for k in kn], fontsize=7)
a2.invert_yaxis(); a2.set_xscale("log"); a2.set_xlabel("georeferenced coverage (km², log)")
a2.set_title("(b)  Coverage area (where reported)", fontsize=9)
for i, k in enumerate(kn):
    a2.text(k[1]*1.15, i, f"{k[1]:,.1f}", va="center", fontsize=6.8)
a2.set_xlim(1, 3000)
fig.savefig(f"{FIG}/f2_datasets.png"); plt.close(fig)

# ============================================================ FIG 3 : model cost
M = [("DenseUAV-ViT\n512-d", 21.86, 4.241, 95.3, 5.99, C["blue"]),
     ("MixVPR\n512-d", 10.09, 8.105, 226.7, 10.36, C["teal"]),
     ("MixVPR\n4096-d", 10.88, 8.421, 241.2, 10.72, C["amber"]),
     ("CosPlace\nRN101 2048-d", 46.70, 17.456, 553.5, 20.92, C["red"])]
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
x = np.arange(len(M)); nm = [m[0] for m in M]
ax[0].bar(x, [m[1] for m in M], color=[m[5] for m in M], edgecolor=C["grey"], lw=0.5)
ax[0].set_ylabel("million parameters"); ax[0].set_title("(a)  Model size", fontsize=9)
for i, m in enumerate(M):
    ax[0].text(i, m[1]+1, f"{m[1]:.1f}", ha="center", fontsize=7)
ax[1].bar(x, [m[2] for m in M], color=[m[5] for m in M], edgecolor=C["grey"], lw=0.5)
ax[1].set_ylabel("GMACs / image"); ax[1].set_title("(b)  Compute per image", fontsize=9)
for i, m in enumerate(M):
    ax[1].text(i, m[2]+0.4, f"{m[2]:.1f}", ha="center", fontsize=7)
w = 0.38
ax[2].bar(x-w/2, [m[3] for m in M], w, color=C["grey"], edgecolor="k", lw=0.4, label="i7-7600U CPU (measured)")
ax[2].bar(x+w/2, [m[4] for m in M], w, color=C["green"], edgecolor="k", lw=0.4, label="Orin Nano Super FP16 (proj.)")
ax[2].set_yscale("log"); ax[2].set_ylabel("latency per image (ms, log)")
ax[2].set_title("(c)  Encode latency", fontsize=9)
ax[2].axhline(100, color=C["red"], ls="--", lw=0.9)
ax[2].text(3.4, 112, "10 Hz budget", color=C["red"], fontsize=6.5, ha="right")
ax[2].legend(fontsize=6, loc="lower left")
for a in ax:
    a.set_xticks(x); a.set_xticklabels(nm, fontsize=6.5)
fig.tight_layout()
fig.savefig(f"{FIG}/f3_models.png"); plt.close(fig)

# ============================================================ FIG 4 : Jetson boards
B = [("Jetson Nano\n(legacy)", 0.236, 10, 199, 212.01, True),
     ("Xavier NX\n16 GB", 10.5, 20, 699, 14.30, False),
     ("Orin Nano\n4 GB Super", 17.0, 25, 349, 10.64, False),
     ("Orin Nano\n8 GB Super", 33.5, 25, 399, 5.99, False),
     ("Orin NX\n8 GB Super", 58.5, 40, 649, 4.52, False),
     ("Orin NX\n16 GB Super", 78.5, 40, 899, 3.91, False),
     ("AGX Orin\n32 GB", 100.0, 60, 1799, 3.46, False),
     ("AGX Orin\n64 GB", 137.5, 75, 2999, 2.95, False),
     ("AGX Thor\nT5000", 1035.0, 130, 5499, 1.55, False)]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(wspace=0.34))
x = np.arange(len(B))
col = [C["light"] if b[5] else (C["green"] if "Orin Nano\n8" in b[0] else C["blue"]) for b in B]
ax[0].bar(x, [b[1] for b in B], color=col, edgecolor=C["grey"], lw=0.5)
ax[0].set_yscale("log"); ax[0].set_ylabel("dense INT8 TOPS (log)")
ax[0].set_title("(a)  AI throughput", fontsize=9)
for i, b in enumerate(B):
    ax[0].text(i, b[1]*1.3, f"{b[1]:g}", ha="center", fontsize=6.2)
ax[0].set_ylim(0.1, 6000)
ax[0].set_xticks(x)
ax[0].set_xticklabels([b[0].replace("\n", " ") for b in B], fontsize=6, rotation=38, ha="right")
ax[0].legend(handles=[Patch(fc=C["green"], ec="k", lw=.4, label="recommended"),
                      Patch(fc=C["light"], ec="k", lw=.4, label="end-of-life")],
             fontsize=6.2, loc="upper left")

tot = [4.0 + b[4]*4 + 0.26 + 0.3 for b in B]
sc = ax[1].scatter([b[3] for b in B], tot, s=[b[2]*4.2 for b in B],
                   c=[b[1] for b in B], cmap="viridis", norm=matplotlib.colors.LogNorm(),
                   edgecolor="k", lw=0.6, zorder=3)
OFF = {"Jetson Nano (legacy)": (8, -4), "Xavier NX 16 GB": (8, 2),
       "Orin Nano 4 GB Super": (10, 6), "Orin Nano 8 GB Super": (2, -16),
       "Orin NX 8 GB Super": (-46, 6), "Orin NX 16 GB Super": (4, 11),
       "AGX Orin 32 GB": (-6, -17), "AGX Orin 64 GB": (2, 11),
       "AGX Thor T5000": (-58, -20)}
for b, t in zip(B, tot):
    nm = b[0].replace("\n", " ")
    ax[1].annotate(nm, (b[3], t), fontsize=6, xytext=OFF[nm], textcoords="offset points")
ax[1].axhline(100, color=C["red"], ls="--", lw=1.0)
ax[1].text(6000, 106, "10 Hz", color=C["red"], fontsize=7, ha="right")
ax[1].axhline(200, color=C["amber"], ls="--", lw=1.0)
ax[1].text(6000, 212, "5 Hz", color=C["amber"], fontsize=7, ha="right")
ax[1].set_yscale("log")
ax[1].set_xlabel("module list price, 2026 (USD)")
ax[1].set_ylabel("end-to-end fix latency (ms, log)")
ax[1].set_title("(b)  Cost vs. latency\nDenseUAV-ViT, 4 rotations, 3,054-tile map", fontsize=9)
ax[1].set_ylim(5, 2000); ax[1].set_xlim(0, 6300)
ax[1].set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000])
ax[1].tick_params(axis="x", labelsize=7)
ax[1].text(0.02, 0.03, "bubble area ∝ module TDP", transform=ax[1].transAxes, fontsize=6,
           color=C["grey"], style="italic")
cb = fig.colorbar(sc, ax=ax[1], pad=0.02); cb.set_label("INT8 TOPS", fontsize=7)
cb.ax.tick_params(labelsize=6)
fig.savefig(f"{FIG}/f4_boards.png"); plt.close(fig)

# ============================================================ FIG 5 : FAISS scaling
N = np.array([1554, 3054, 7204, 12215, 45028, 180112, 1000000], float)
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
for dim, kind, c, ls, lab in [(512, "flat", C["blue"], "-", "D=512 flat"),
                              (512, "hnsw", C["teal"], "-", "D=512 HNSW"),
                              (512, "ivfpq", C["green"], "-", "D=512 IVF-PQ64"),
                              (4096, "flat", C["red"], "--", "D=4096 flat"),
                              (4096, "ivfpq", C["amber"], "--", "D=4096 IVF-PQ64")]:
    bpv = {"flat": dim*4, "hnsw": dim*4+271, "ivfpq": 122}[kind]
    ax[0].loglog(N, N*bpv/1e6, ls, color=c, marker="o", ms=2.6, lw=1.3, label=lab)
ax[0].axhline(8000, color=C["grey"], ls=":", lw=1.0)
ax[0].text(1700, 9200, "8 GB (Orin Nano RAM)", fontsize=6.2, color=C["grey"])
ax[0].set_xlabel("gallery size N (tiles)"); ax[0].set_ylabel("index size (MB, log)")
ax[0].set_title("(a)  Index memory footprint", fontsize=9)
ax[0].legend(fontsize=6.2, loc="upper left")

meas = {"D=512 flat": ([1554, 7204, 45028], [0.272, 1.235, 45.469], C["blue"], "-"),
        "D=512 HNSW": ([1554, 7204, 45028], [0.227, 1.879, 3.535], C["teal"], "-"),
        "D=512 IVF-PQ": ([7204, 45028], [0.170, 0.197], C["green"], "-"),
        "D=2048 flat": ([1554, 7204, 45028], [3.142, 8.998, 138.283], C["purple"], "--"),
        "D=4096 flat": ([1554, 7204, 45028], [4.700, 30.853, 232.387], C["red"], "--")}
for lab, (nn, tt, c, ls) in meas.items():
    ax[1].loglog(nn, tt, ls, color=c, marker="s", ms=3, lw=1.3, label=lab)
ax[1].axhline(100, color=C["red"], ls=":", lw=1.0)
ax[1].text(1700, 115, "10 Hz whole-pipeline budget", fontsize=6.2, color=C["red"])
ax[1].set_xlabel("gallery size N (tiles)"); ax[1].set_ylabel("search latency (ms, log)")
ax[1].set_title("(b)  Measured search latency, i7-7600U, k=10", fontsize=9)
ax[1].legend(fontsize=6.2, loc="upper left")
ax[1].axvspan(2800, 3300, color=C["purple"], alpha=0.12)
ax[1].text(3054, 0.06, "Ardabil\n3,054", fontsize=6, ha="center", color=C["purple"])
fig.tight_layout()
fig.savefig(f"{FIG}/f5_faiss.png"); plt.close(fig)

# ============================================================ FIG 6 : latency budget stacks
CPU_CFG = [
    ("i7-7600U · CosPlace-2048 ×4", 12.0, 2214.0, 33.95),
    ("i7-7600U · MixVPR-4096 ×4",   12.0,  964.8, 48.92),
    ("i7-7600U · MixVPR-512 ×4",    12.0,  906.8,  0.42),
    ("i7-7600U · DenseUAV-ViT ×4",  12.0,  381.2,  0.42),
    ("i7-7600U · MixVPR-4096 ×1",   12.0,  241.2,  9.24),
    ("i7-7600U · DenseUAV-ViT ×1",  12.0,   95.3,  0.17),
]
JET_CFG = [
    ("Orin Nano 8GB · CosPlace ×4",     4.0, 83.7, 11.32),
    ("Orin Nano 8GB · MixVPR-4096 ×4",  4.0, 42.9, 16.31),
    ("Orin Nano 8GB · DenseUAV-ViT ×4", 4.0, 24.0,  0.26),
    ("Orin NX 16GB · DenseUAV-ViT ×4",  4.0, 15.6,  0.26),
    ("AGX Orin 64GB · DenseUAV-ViT ×4", 4.0, 11.8,  0.26),
    ("Orin Nano 8GB · DenseUAV-ViT ×1", 4.0,  6.0,  0.11),
]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw=dict(wspace=0.62))
for ax, CFG, ttl, xmax in [(axs[0], CPU_CFG, "(a)  Laptop — Intel i7-7600U, PyTorch CPU FP32", 3050),
                           (axs[1], JET_CFG, "(b)  Jetson — TensorRT FP16 (projected)", 132)]:
    y = np.arange(len(CFG))
    pre = np.array([c[1] for c in CFG]); enc = np.array([c[2] for c in CFG])
    srch = np.array([c[3] for c in CFG]); fuse = np.full(len(CFG), 0.3)
    ax.barh(y, pre, color="#a0aec0", edgecolor="k", lw=0.4, label="decode + resize")
    ax.barh(y, enc, left=pre, color=C["blue"], edgecolor="k", lw=0.4, label="neural encode")
    ax.barh(y, srch, left=pre+enc, color=C["amber"], edgecolor="k", lw=0.4, label="FAISS search")
    ax.barh(y, fuse, left=pre+enc+srch, color=C["green"], edgecolor="k", lw=0.4, label="geo fusion")
    tot = pre+enc+srch+fuse
    for i, t in enumerate(tot):
        hz = 1000/t
        lab = f"{t:.1f} ms · {hz:.0f} Hz" if hz >= 10 else f"{t:.1f} ms · {hz:.1f} Hz"
        inside = t > 0.68*xmax
        ax.text(t - xmax*0.015 if inside else t + xmax*0.015, i, lab, va="center",
                ha="right" if inside else "left", fontsize=6.6, fontweight="bold",
                color="white" if inside else (C["red"] if t > 100 else C["green"]))
    ax.set_yticks(y); ax.set_yticklabels([c[0] for c in CFG], fontsize=6.6)
    ax.invert_yaxis(); ax.set_xlim(0, xmax)
    ax.set_xlabel("latency (ms)"); ax.set_title(ttl, fontsize=8.5)
    ax.axvline(100, color=C["red"], ls="--", lw=1.0)
    if xmax > 500:
        ax.axvline(200, color=C["amber"], ls="--", lw=1.0)
        ax.text(320, -0.85, "5 Hz", color=C["amber"], fontsize=6.5)
    ax.text(105 if xmax < 500 else -180, -0.85, "10 Hz", color=C["red"], fontsize=6.5)
    ax.set_ylim(len(CFG)-0.4, -1.15)
axs[1].legend(fontsize=6.6, ncol=4, loc="upper center", bbox_to_anchor=(-0.28, -0.24),
              framealpha=0.95, columnspacing=1.1, handlelength=1.3)
fig.suptitle("End-to-end single-fix latency budget — Ardabil map (3,054 tiles)",
             fontsize=10, fontweight="bold", y=1.02)
fig.savefig(f"{FIG}/f6_budget.png"); plt.close(fig)

# ============================================================ FIG 7 : Ardabil sizing
fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.1))
# (a) tile counts
tl = [("0.3 m\n0 %", 764), ("0.3 m\n50 %", 3054), ("0.3 m\n75 %", 12215),
      ("0.5 m\n0 %", 275), ("0.5 m\n50 %", 1100), ("0.5 m\n75 %", 4398)]
cc = [C["purple"] if t[1] == 3054 else C["light"] for t in tl]
ax[0].bar(range(6), [t[1] for t in tl], color=cc, edgecolor=C["grey"], lw=0.5)
ax[0].set_yscale("log"); ax[0].set_xticks(range(6))
ax[0].set_xticklabels([t[0] for t in tl], fontsize=6.2)
ax[0].set_ylabel("reference tiles (log)")
ax[0].set_title("(a)  Ardabil map, 18.01 km²\nGSD / overlap", fontsize=8.5)
for i, t in enumerate(tl):
    ax[0].text(i, t[1]*1.25, f"{t[1]:,}", ha="center", fontsize=6)
ax[0].set_ylim(100, 40000)

# (b) storage stack for the chosen 3054-tile map
comp = ["JPEG tiles (q85)", "descriptors fp32 D512", "FAISS flat D512",
        "FAISS IVF-PQ D512", "ViT weights FP16", "FAISS flat D4096"]
vals = [3054*26.2/1024, 3054*2048/1e6, 3054*2048/1e6, 3054*122/1e6, 41.7, 3054*16384/1e6]
ax[1].bar(range(6), vals, color=[C["teal"], C["blue"], C["amber"], C["green"], C["grey"], C["red"]],
          edgecolor=C["grey"], lw=0.5)
ax[1].set_xticks(range(6)); ax[1].set_xticklabels(comp, fontsize=6, rotation=38, ha="right")
ax[1].set_ylabel("MB"); ax[1].set_title("(b)  On-board storage,\n3,054-tile Ardabil map", fontsize=8.5)
for i, v in enumerate(vals):
    ax[1].text(i, v+2.0, f"{v:.1f}", ha="center", fontsize=6)
ax[1].set_ylim(0, 96)

# (c) acquisition cost: survey vs satellite tiling
steps = [20, 50, 100, 150, 200]
hrs = [16.7, 6.7, 3.3, 2.2, 1.7]
ax[2].plot(steps, hrs, "o-", color=C["red"], lw=1.4, ms=4, label="UAV survey flight hours")
ax[2].axhline(0.15, color=C["green"], lw=1.4, ls="-")
ax[2].text(60, 0.20, "satellite tiling: ~9 min download\n+ 4.3 min index build (CPU)",
           fontsize=6.3, color=C["green"])
ax[2].set_yscale("log"); ax[2].set_xlabel("waypoint spacing (m)")
ax[2].set_ylabel("hours to build map (log)")
ax[2].set_title("(c)  Map-construction effort\nfor Ardabil", fontsize=8.5)
for s, h in zip(steps, hrs):
    ax[2].text(s, h*1.2, f"{h:.1f} h", fontsize=6, ha="center")
ax[2].set_ylim(0.08, 60)
ax[2].legend(fontsize=6, loc="upper right")
fig.tight_layout()
fig.savefig(f"{FIG}/f7_ardabil.png"); plt.close(fig)

# ============================================================ FIG 8 : accuracy from artifacts
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.7))
mm = ["MixVPR-4096\n(200 refs)", "MixVPR-4096\n(1554 refs)", "DenseUAV-ViT\n(200 refs)",
      "DenseUAV-ViT\n(1554 refs)"]
top1 = [171.4, 28.8, 20.0, 20.0]
fus = [151.3, 189.8, 5.5, 5.5]
x = np.arange(4); w = 0.38
ax[0].bar(x-w/2, top1, w, color=C["light"], edgecolor="k", lw=0.4, label="Top-1 match error")
ax[0].bar(x+w/2, fus, w, color=C["blue"], edgecolor="k", lw=0.4, label="fused (Top-5) error")
ax[0].set_yscale("log"); ax[0].set_ylabel("localisation error (m, log)")
ax[0].set_xticks(x); ax[0].set_xticklabels(mm, fontsize=6)
ax[0].axhline(19.7, color=C["red"], ls="--", lw=0.9)
ax[0].text(3.45, 22, "map cell = 19.7 m", fontsize=6, color=C["red"], ha="right")
ax[0].set_title("(a)  Measured localisation error on DenseUAV", fontsize=8.5)
ax[0].legend(fontsize=6)
for i, (a_, b_) in enumerate(zip(top1, fus)):
    ax[0].text(i-w/2, a_*1.12, f"{a_:.0f}", ha="center", fontsize=6)
    ax[0].text(i+w/2, b_*1.12, f"{b_:.1f}", ha="center", fontsize=6)

# rotation sensitivity / cost
rots = [1, 4]
enc_cpu = [95.3, 381.2]; enc_orin = [5.99, 23.96]
ax[1].plot(rots, enc_cpu, "o-", color=C["grey"], lw=1.5, ms=5, label="i7-7600U CPU (measured)")
ax[1].plot(rots, enc_orin, "s-", color=C["green"], lw=1.5, ms=5, label="Orin Nano Super (proj.)")
ax[1].axhline(100, color=C["red"], ls="--", lw=1.0)
ax[1].text(2.5, 112, "10 Hz budget", fontsize=6.5, color=C["red"])
ax[1].set_yscale("log"); ax[1].set_xticks([1, 4]); ax[1].set_xlim(0.6, 4.4)
ax[1].set_xlabel("query rotations evaluated (unknown-heading robustness)")
ax[1].set_ylabel("encode latency (ms, log)")
ax[1].set_title("(b)  Cost of the ×4 rotation search", fontsize=8.5)
ax[1].legend(fontsize=6.5, loc="center right")
for r, v in zip(rots, enc_cpu):
    ax[1].text(r, v*1.2, f"{v:.0f} ms", fontsize=6.3, ha="center")
for r, v in zip(rots, enc_orin):
    ax[1].text(r, v*0.62, f"{v:.1f} ms", fontsize=6.3, ha="center")
fig.tight_layout()
fig.savefig(f"{FIG}/f8_accuracy.png"); plt.close(fig)

# ============================================================ FIG 9 : system pipeline
fig, ax = plt.subplots(figsize=(7.4, 3.0))
ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 44)
def box(x, y, w, h, t, c, fs=7, tc="white"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=c, edgecolor="black", lw=0.7,
                           joinstyle="round", alpha=0.95))
    ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold")
def arrow(x1, y1, x2, y2, lab=""):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#2d3748"))
    if lab:
        ax.text((x1+x2)/2, (y1+y2)/2+1.3, lab, ha="center", fontsize=6, color="#2d3748")

ax.text(1, 40.5, "OFFLINE  (ground station / laptop — once per city)", fontsize=8,
        fontweight="bold", color=C["grey"])
box(1, 30, 17, 7, "Ortho satellite\nmap of Ardabil\n0.3 m GSD", C["teal"])
box(21, 30, 17, 7, "Tiling\n512×512, 50 % ovl\n→ 3,054 tiles", C["teal"])
box(41, 30, 17, 7, "DenseUAV-ViT\nencoder\n4.24 GMACs", C["blue"])
box(61, 30, 17, 7, "L2-normalised\n512-d descriptors\n6.3 MB", C["blue"])
box(81, 30, 17, 7, "FAISS index\nIVF-PQ 0.4 MB\nor flat 6.3 MB", C["amber"])
for x in (18, 38, 58, 78):
    arrow(x, 33.5, x+3, 33.5)
ax.text(1, 25.5, "→ total offline cost:  ~9 min tile download  +  4.3 min encode on the i7-7600U laptop  "
                 "(0.2 min on an Orin Nano)", fontsize=7, color=C["grey"], style="italic")

ax.text(1, 20.5, "ONLINE  (on-board Jetson — every frame)", fontsize=8,
        fontweight="bold", color=C["grey"])
box(1, 9, 17, 7, "UAV nadir\ncamera frame\n1440×1080", C["purple"])
box(21, 9, 17, 7, "decode + resize\n224×224\n4 ms", "#718096")
box(41, 9, 17, 7, "DenseUAV-ViT\nTensorRT FP16\n6.0 ms ×R", C["blue"])
box(61, 9, 17, 7, "FAISS top-K\nsearch\n0.26 ms", C["amber"])
box(81, 9, 17, 7, "weighted geo\nfusion → lat/lon\n0.3 ms", C["green"])
for x in (18, 38, 58, 78):
    arrow(x, 12.5, x+3, 12.5)
ax.text(1, 4.4, "→ single fix:  10.4 ms (96 Hz) with 1 orientation  |  28.5 ms (35 Hz) with 4 orientations "
                "— on a $399 Orin Nano 8 GB Super at 25 W", fontsize=7, color=C["green"],
        fontweight="bold")
ax.text(1, 1.4, "→ same pipeline on the i7-7600U laptop:  107.8 ms (9.3 Hz) / 393.9 ms (2.5 Hz)",
        fontsize=7, color=C["red"])
fig.savefig(f"{FIG}/f9_pipeline.png"); plt.close(fig)

print("figures written to", FIG)
for f in sorted(os.listdir(FIG)):
    print("  ", f, os.path.getsize(os.path.join(FIG, f))//1024, "KB")
