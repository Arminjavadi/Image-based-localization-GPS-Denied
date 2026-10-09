"""Figures and tables for the scale-normalisation report (docs/Scale_Normalization_Report.md).

Reads what scripts/visloc_scale.py wrote to artifacts/visloc/scale/ and writes PNGs to
docs/report_figs/scale/ plus a markdown table digest to artifacts/visloc/scale/tables.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCALE = Path("artifacts/visloc/scale")
FIGS = Path("docs/report_figs/scale")
REGIONS = ["05", "10", "06", "11"]
POLICY_LABEL = {
    "rot90": "heading, 90° step, full square (current)",
    "north": "exact north-align, full square",
    "blind3": "blind 3-scale search (3× encodes)",
    "agl": "AGL crop (1 encode)",
    "agl_prior": "AGL crop, DEM at prior ±300 m, baro bias",
    "oracle_grid": "oracle: best grid scale per frame (uses GT)",
}
COLORS = {"rot90": "#9aa0a6", "north": "#5f6368", "blind3": "#c58af9", "agl": "#1a73e8",
          "agl_prior": "#8ab4f8", "oracle_grid": "#fbbc04"}

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def fig_agl() -> None:
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(12, 2.8))
    for ax, region in zip(axes, REGIONS):
        a = pd.read_csv(SCALE / f"agl_{region}.csv")
        x = np.arange(len(a))
        ax.plot(x, a["height"], color="#5f6368", lw=1, label="flight-log altitude (ASL)")
        ax.plot(x, a["terrain_m"], color="#188038", lw=1, label="terrain (GLO-30)")
        ax.fill_between(x, a["terrain_m"], a["height"], color="#1a73e8", alpha=0.12, label="AGL")
        ax.axvspan(0, min(144, len(a)), color="#fbbc04", alpha=0.10, lw=0)
        ax.set_title(f"r{region}: AGL {a['agl_m'].iloc[:144].median():.0f} m "
                     f"(range {a['agl_m'].iloc[:144].min():.0f}–{a['agl_m'].iloc[:144].max():.0f})")
        ax.set_xlabel("frame")
    axes[0].set_ylabel("metres above sea level")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Altitude above sea level is not height above ground (shaded: the 144 evaluated frames)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "s1_agl_profiles.png", bbox_inches="tight")
    plt.close(fig)


def fig_calibration(model: str = "megaloc") -> None:
    cal = pd.read_csv(SCALE / f"calibration_{model}.csv")
    cal["region"] = cal["region"].astype(str).str.zfill(2)
    summary = json.loads((SCALE / f"calibration_{model}.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axes[0]
    for camera, marker in (("A", "o"), ("B", "s")):
        c = cal[(cal["camera"] == camera) & ~cal["edge"].astype(bool)]
        for region, g in c.groupby("region"):
            ax.scatter(g["agl_m"], g["k"], marker=marker, s=18, alpha=0.75, label=f"r{region} ({camera})")
        if camera in summary["cameras"]:
            k = summary["cameras"][camera]["k_median"]
            ax.axhline(k, ls="--", lw=1, color="k" if camera == "A" else "#d93025")
            ax.text(ax.get_xlim()[1] if False else c["agl_m"].max(), k, f" k_{camera}={k:.2f}", va="bottom", fontsize=8)
    ax.set_xlabel("height above ground (m)")
    ax.set_ylabel("k = full-frame ground width / AGL")
    ax.set_title("Per-frame estimates: noisy (encoder scan at the true position)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    # what the method uses: one k per region (peak of the mean curve), from two encoders
    agl_med = {r: float(pd.read_csv(SCALE / f"agl_{r}.csv")["agl_m"].median()) for r in summary["regions"]}
    for mdl, marker, color in (("megaloc", "o", "#1a73e8"), ("denseuav-vit", "^", "#e37400")):
        path = SCALE / f"calibration_{mdl}.json"
        if not path.exists():
            continue
        regs = json.loads(path.read_text())["regions"]
        for r, v in regs.items():
            k = v["k_outside_eval"] or v["k"]
            face = color if v["camera"] == "B" else "white"
            ax.scatter(agl_med[r], k, marker=marker, s=46, color=face, edgecolors=color, zorder=3,
                       label=f"{mdl}" if r == next(iter(regs)) else None)
            if mdl == "megaloc":
                ax.annotate(f"r{r} ({v['camera']})", (agl_med[r], k), textcoords="offset points",
                            xytext=(6, -3), fontsize=7)
    kb = summary["cameras"]["B"]["k"]
    ax.axhspan(kb * 0.92, kb * 1.08, color="#d93025", alpha=0.08, lw=0)
    ax.axhline(kb, color="#d93025", lw=1, ls="--")
    ax.set_xlabel("region median height above ground (m)")
    ax.set_ylabel("region k (frames outside the eval set)")
    ax.set_title("Region k: camera B (filled) 0.88–1.05 over 308–778 m AGL")
    ax.legend(fontsize=7, loc="upper right")
    slope = None
    ok = cal[(~cal["edge"].astype(bool)) & (cal["camera"] == "B")]
    if len(ok) > 5:
        slope = float(np.polyfit(np.log(ok["agl_m"]), np.log(ok["best_size_m"] / ok["side_px"] * ok["width"]), 1)[0])
    fig.tight_layout()
    fig.savefig(FIGS / "s2_calibration.png", bbox_inches="tight")
    plt.close(fig)
    return slope


def fig_curves(model: str = "megaloc", n: int = 6) -> None:
    cal = pd.read_csv(SCALE / f"calibration_{model}.csv")
    cal = cal[~cal["edge"].astype(bool)]
    pick = pd.concat([g.iloc[[len(g) // 2]] for _, g in cal.groupby("region")]).head(n)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for _, r in pick.iterrows():
        sizes = np.array([float(v) for v in r["sizes"].split("|")])
        sims = np.array([float(v) for v in r["sims"].split("|")])
        rel = sizes / r["best_size_m"]
        ax.plot(rel, sims, marker="o", ms=3, lw=1, label=f"r{str(r['region']).zfill(2)} {r['stem']} (AGL {r['agl_m']:.0f} m)")
    ax.set_xscale("log", base=2)
    ax.axvline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("satellite patch ground size / best size")
    ax.set_ylabel("cosine similarity to the frame")
    ax.set_title("Encoder similarity peaks at one ground scale")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIGS / "s3_scale_curves.png", bbox_inches="tight")
    plt.close(fig)


def load_evals(model: str) -> dict:
    out = {}
    for region in REGIONS:
        p = SCALE / f"eval_{region}__{model}.json"
        if p.exists():
            out[region] = json.loads(p.read_text())
    return out


def fig_results(model: str, center: str, metric: str = "top1@100") -> None:
    evals = load_evals(model)
    if not evals:
        return
    blocks = []
    for region, e in evals.items():
        for tiles, b in e["tiles"].items():
            blocks.append((f"r{region}\n{tiles.split('_', 1)[1]}", b["centers"].get(center, {})))
    policies = [p for p in POLICY_LABEL if any(p in b for _, b in blocks)]
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(blocks)), 3.6))
    width = 0.8 / len(policies)
    for j, p in enumerate(policies):
        vals = [100 * b.get(p, {}).get(metric, np.nan) for _, b in blocks]
        xs = np.arange(len(blocks)) + (j - (len(policies) - 1) / 2) * width
        bars = ax.bar(xs, vals, width, color=COLORS[p], label=POLICY_LABEL[p],
                      hatch="//" if p == "oracle_grid" else None, edgecolor="white", lw=0.5)
        for x, v in zip(xs, vals):
            if np.isfinite(v):
                ax.text(x, v + 0.8, f"{v:.0f}", ha="center", fontsize=6)
    ax.set_xticks(np.arange(len(blocks)))
    ax.set_xticklabels([lab for lab, _ in blocks], fontsize=8)
    ax.set_ylabel(f"{metric.replace('@', ' within ')} m (%)")
    ax.set_title(f"{model}: query-scale policies (centering: {center})")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    safe = center.replace("+", "_")
    fig.savefig(FIGS / f"s4_results_{model}_{safe}.png", bbox_inches="tight")
    plt.close(fig)


def fig_error_vs_agl(model: str, region: str, tiles: str) -> None:
    p = SCALE / f"eval_{region}__{model}_per_query.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for pol, color in (("north", COLORS["north"]), ("agl", COLORS["agl"])):
        col = f"{tiles}__map+flight__{pol}__top1_err"
        if col not in d:
            continue
        ok = d[col] <= 100
        ax.scatter(d["agl_m"][ok], d[col][ok].clip(upper=3000), s=12, color=color, alpha=0.8, label=f"{pol}: ≤100 m")
        ax.scatter(d["agl_m"][~ok], d[col][~ok].clip(upper=3000), s=12, facecolors="none", edgecolors=color,
                   alpha=0.6, label=f"{pol}: >100 m")
    ax.set_yscale("log")
    ax.axhline(100, color="k", lw=0.8, ls=":")
    ax.set_xlabel("height above ground (m)")
    ax.set_ylabel("top-1 error (m, log)")
    ax.set_title(f"r{region} {tiles}: per-frame error vs height above ground")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / f"s5_error_vs_agl_r{region}.png", bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity(model: str, center: str) -> None:
    evals = load_evals(model)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    drew = False
    for region, e in evals.items():
        for tiles, b in e["tiles"].items():
            res = b["centers"].get(center, {})
            pts = [(0.0, res.get("agl", {}).get("top1@100"))]
            for name, m in res.items():
                if name.startswith("agl") and name[3:4] in "+-" and name.endswith("%"):
                    pts.append((float(name[3:-1]) / 100.0, m["top1@100"]))
            pts = sorted(p for p in pts if p[1] is not None)
            if len(pts) > 1:
                ax.plot([100 * p[0] for p in pts], [100 * p[1] for p in pts], marker="o",
                        label=f"r{region} {tiles.split('_', 1)[1]}")
                if "north" in res:
                    ax.axhline(100 * res["north"]["top1@100"], ls=":", lw=0.8)
                drew = True
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("AGL error injected (%)")
    ax.set_ylabel("top-1 within 100 m (%)")
    ax.set_title("Sensitivity to height error (dotted: no scale normalisation)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / f"s6_sensitivity_{model}.png", bbox_inches="tight")
    plt.close(fig)


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired successes: (a-only wins, b-only wins, p)."""
    wins_a = int(np.sum(a & ~b))
    wins_b = int(np.sum(~a & b))
    n = wins_a + wins_b
    if n == 0:
        return wins_a, wins_b, 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(wins_a, wins_b) + 1)) / 2 ** n
    return wins_a, wins_b, min(1.0, 2 * tail)


def paired_tests(models: list[str], threshold: float = 100.0,
                 pairs=(("agl", "north"), ("agl", "blind3"), ("agl", "rot90"), ("agl_prior", "agl"))) -> str:
    lines = [f"\n### Paired tests (top-1 within {threshold:.0f} m, exact McNemar)\n",
             "| Model | Region / map | Centering | Comparison | Δ pts | wins / losses | p |",
             "|---|---|---|---|---:|---:|---:|"]
    for model in models:
        for region in REGIONS:
            p = SCALE / f"eval_{region}__{model}_per_query.csv"
            e = SCALE / f"eval_{region}__{model}.json"
            if not (p.exists() and e.exists()):
                continue
            d = pd.read_csv(p)
            for tiles in json.loads(e.read_text())["tiles"]:
                for center in ("off", "map+flight"):
                    for x, y in pairs:
                        cx, cy = f"{tiles}__{center}__{x}__top1_err", f"{tiles}__{center}__{y}__top1_err"
                        if cx not in d or cy not in d:
                            continue
                        sx, sy = d[cx].to_numpy() <= threshold, d[cy].to_numpy() <= threshold
                        wa, wb, pv = mcnemar(sx, sy)
                        delta = 100 * (sx.mean() - sy.mean())
                        lines.append(f"| {model} | r{region} {tiles} | {center} | {x} vs {y} | "
                                     f"{delta:+.1f} | {wa} / {wb} | {pv:.3f} |")
    return "\n".join(lines)


BASE_MAP = {"05": "r05_t250", "10": "r10_t200", "06": "r06_t200", "11": "r11_t250"}


def physics_map(region: str) -> str | None:
    plan = SCALE / "plan.json"
    if not plan.exists():
        return None
    entry = json.loads(plan.read_text()).get(region)
    return entry["tiles"] if entry and entry.get("rule") == "short-median" else None


def headline(model: str, center: str = "map+flight", metric: str = "top1@100") -> str:
    """Query-side vs map-side normalisation, one row per region, plus paired tests."""
    rows = ["| Region | Map so far | AGL-sized map | 90° view, map so far | 90° view, AGL-sized map | "
            "per-frame map select | per-frame map select, DEM at prior | AGL crop, map so far | "
            "blind 3-scale, map so far | oracle scale |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    tests = ["| Region | Comparison | Δ pts | wins / losses | p |", "|---|---|---:|---:|---:|"]
    pooled: dict[str, list] = {}
    fig_rows = []
    for region in REGIONS:
        e_path = SCALE / f"eval_{region}__{model}.json"
        q_path = SCALE / f"eval_{region}__{model}_per_query.csv"
        if not e_path.exists():
            continue
        e = json.loads(e_path.read_text())
        base, phys = BASE_MAP[region], physics_map(region)
        if base not in e["tiles"]:
            continue
        b = e["tiles"][base]["centers"][center]
        p = e["tiles"].get(phys, {}).get("centers", {}).get(center) if phys else None
        # prefer the centre-aligned version of the AGL-sized map (same tile centres as base)
        al_path = SCALE / f"eval_{region}__{model}__aligned.json"
        al_q = SCALE / f"eval_{region}__{model}__aligned_per_query.csv"
        phys_col_src = (q_path, phys)
        if phys and al_path.exists() and al_q.exists():
            al = json.loads(al_path.read_text())
            if f"{phys}_c" in al["tiles"]:
                p = al["tiles"][f"{phys}_c"]["centers"][center]
                phys_col_src = (al_q, f"{phys}_c")
        ms = e.get("map_select", {})
        sel = ms.get("map_select", {}).get(center)
        selp = ms.get("map_select_prior", {}).get(center)

        def pct(m):
            return "—" if m is None else f"{100 * m[metric]:.1f}"

        rows.append(f"| r{region} | {base.split('_', 1)[1]} | {phys.split('_', 1)[1] if phys else '—'} | "
                    f"{pct(b['rot90'])} | {pct(p['rot90'] if p else None)} | {pct(sel)} | {pct(selp)} | "
                    f"{pct(b['agl'])} | {pct(b['blind3'])} | {pct(b['oracle_grid'])} |")
        gated_val = np.nan
        if q_path.exists():
            dq = pd.read_csv(q_path)
            wq, hq = FRAME[e["camera"]]
            ratio = min(wq, hq) * e["k"] * np.clip(dq["agl_m"].to_numpy(), 100, None) / wq / e["tiles"][base]["tile_m"]
            thr_m = float(metric.split("@")[1])
            g_ok = np.where(ratio > 1.25, dq[f"{base}__{center}__agl__top1_err"] <= thr_m,
                            dq[f"{base}__{center}__rot90__top1_err"] <= thr_m)
            gated_val = float(np.mean(g_ok))
        fig_rows.append((region, {
            "90° view, map so far (current)": b["rot90"][metric],
            "blind 3-scale (3× cost)": b["blind3"][metric],
            "AGL crop, always": b["agl"][metric],
            "AGL crop, gated τ=1.25 (recommended)": gated_val,
            "90° view, AGL-sized map": p["rot90"][metric] if p else np.nan,
        }))
        if q_path.exists():
            d = pd.read_csv(q_path)
            thr = float(metric.split("@")[1])
            ok = lambda col: d[col].to_numpy() <= thr  # noqa: E731
            pairs = [("exact north vs 90° view (map so far)",
                      f"{base}__{center}__north__top1_err", f"{base}__{center}__rot90__top1_err"),
                     ("AGL crop vs uncropped north (query side)",
                      f"{base}__{center}__agl__top1_err", f"{base}__{center}__north__top1_err"),
                     ("blind 3-scale vs uncropped north",
                      f"{base}__{center}__blind3__top1_err", f"{base}__{center}__north__top1_err")]
            if phys:
                src, name = phys_col_src
                dd = pd.read_csv(src)
                col = f"{name}__{center}__rot90__top1_err"
                if col in dd:
                    d[f"__phys__{center}"] = dd[col].to_numpy()
                    pairs.append(("AGL-sized map (centre-aligned) vs map so far (90° view)"
                                  if name.endswith("_c") else "AGL-sized map vs map so far (90° view)",
                                  f"__phys__{center}", f"{base}__{center}__rot90__top1_err"))
            if sel:
                pairs.append(("per-frame map select vs map so far (90° view)",
                              f"map_select__{center}__top1_err", f"{base}__{center}__rot90__top1_err"))
            for label, cx, cy in pairs:
                if cx in d and cy in d:
                    wa, wb, pv = mcnemar(ok(cx), ok(cy))
                    tests.append(f"| r{region} | {label} | {100 * (ok(cx).mean() - ok(cy).mean()):+.1f} | "
                                 f"{wa} / {wb} | {pv:.3f} |")
                    acc = pooled.setdefault(label, [0, 0, 0, []])
                    acc[0] += wa
                    acc[1] += wb
                    acc[2] += len(d)
                    acc[3].append(region)
    if fig_rows:
        labels = list(fig_rows[0][1])
        colors = ["#9aa0a6", "#c58af9", "#f6c26b", "#e37400", "#1a73e8"]
        fig, ax = plt.subplots(figsize=(1.9 * len(fig_rows) + 2, 3.6))
        width = 0.8 / len(labels)
        for j, (lab, col) in enumerate(zip(labels, colors)):
            vals = [100 * r[1][lab] for r in fig_rows]
            xs = np.arange(len(fig_rows)) + (j - (len(labels) - 1) / 2) * width
            ax.bar(xs, vals, width, color=col, label=lab, edgecolor="white", lw=0.5)
            for x, v in zip(xs, vals):
                if np.isfinite(v):
                    ax.text(x, v + 0.8, f"{v:.0f}", ha="center", fontsize=6.5)
        ax.set_xticks(np.arange(len(fig_rows)))
        ax.set_xticklabels([f"r{r}" for r, _ in fig_rows])
        ax.set_ylabel("top-1 within 100 m (%)")
        ax.set_title(f"{model}, centering {center}: scale from altitude — query side vs map side")
        ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
        fig.tight_layout()
        fig.savefig(FIGS / f"s7_headline_{model}_{center.replace('+', '_')}.png", bbox_inches="tight")
        plt.close(fig)
    for label, (wa, wb, n, regs) in pooled.items():
        if len(regs) > 1:
            k = wa + wb
            pv = min(1.0, 2 * sum(math.comb(k, i) for i in range(min(wa, wb) + 1)) / 2 ** k) if k else 1.0
            tests.append(f"| **pooled** ({', '.join('r' + r for r in regs)}) | {label} | "
                         f"{100 * (wa - wb) / n:+.1f} | {wa} / {wb} | {pv:.3f} |")
    return (f"\n### Headline — {model}, centering {center}, {metric}\n\n" + "\n".join(rows)
            + "\n\n" + "\n".join(tests))


FRAME = {"A": (3000, 2000), "B": (3976, 2652)}


def gated(model: str, center: str = "map+flight", taus=(1.1, 1.2, 1.3, 1.5, 1.7), metric_m: float = 100.0) -> str:
    """Crop only when AGL says the frame is much larger than a tile; otherwise keep the
    uncropped 90-degree view. Computed from saved per-frame errors (no re-encoding).
    tau is reported over a range instead of being tuned on these frames."""
    lines = [f"\n### Gated AGL crop — {model}, {center}, top-1 within {metric_m:.0f} m\n",
             "| Region / map | footprint ÷ tile (median) | 90° view | always crop | " +
             " | ".join(f"gate τ={t:g}" for t in taus) + " |",
             "|---|---:|---:|---:|" + "---:|" * len(taus)]
    pooled = {t: [0, 0] for t in ("rot90", "agl", *taus)}
    for region in REGIONS:
        e_path = SCALE / f"eval_{region}__{model}.json"
        q_path = SCALE / f"eval_{region}__{model}_per_query.csv"
        if not (e_path.exists() and q_path.exists()):
            continue
        e = json.loads(e_path.read_text())
        d = pd.read_csv(q_path)
        w, h = FRAME[e["camera"]]
        foot = min(w, h) * e["k"] * np.clip(d["agl_m"].to_numpy(), 100, None) / w
        for tiles, b in e["tiles"].items():
            r90 = f"{tiles}__{center}__rot90__top1_err"
            agl = f"{tiles}__{center}__agl__top1_err"
            if r90 not in d or agl not in d:
                continue
            ratio = foot / b["tile_m"]
            ok_r, ok_a = d[r90].to_numpy() <= metric_m, d[agl].to_numpy() <= metric_m
            cells = [f"{100 * ok_r.mean():.1f}", f"{100 * ok_a.mean():.1f}"]
            if tiles == BASE_MAP.get(region):
                pooled["rot90"][0] += ok_r.sum(); pooled["rot90"][1] += len(d)
                pooled["agl"][0] += ok_a.sum(); pooled["agl"][1] += len(d)
            for t in taus:
                ok_g = np.where(ratio > t, ok_a, ok_r)
                cells.append(f"{100 * ok_g.mean():.1f}")
                if tiles == BASE_MAP.get(region):
                    pooled[t][0] += ok_g.sum(); pooled[t][1] += len(d)
            lines.append(f"| r{region} {tiles} | {np.median(ratio):.2f} | " + " | ".join(cells) + " |")
    if pooled["rot90"][1]:
        lines.append("| **mean over maps so far** | | " + " | ".join(
            f"**{100 * pooled[k][0] / pooled[k][1]:.1f}**" for k in ("rot90", "agl", *taus)) + " |")
    return "\n".join(lines)


def aligned(models: list[str], metric_m: float = 100.0) -> str:
    """Map-side comparison on centre-aligned maps (eval tag __aligned): 90-degree view on each
    map vs the original map, paired McNemar, both centering modes."""
    lines = [f"\n### Centre-aligned maps (90° view, top-1 within {metric_m:.0f} m)\n",
             "| Model | Region | Map | Tile | Nearest centre | `map+flight` | Δ vs original (W/L, p) | off | Δ vs original (W/L, p) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for model in models:
        for region in REGIONS:
            e_path = SCALE / f"eval_{region}__{model}__aligned.json"
            q_path = SCALE / f"eval_{region}__{model}__aligned_per_query.csv"
            if not (e_path.exists() and q_path.exists()):
                continue
            e = json.loads(e_path.read_text())
            d = pd.read_csv(q_path)
            base = BASE_MAP[region]
            for tiles, b in e["tiles"].items():
                cells = []
                for center in ("map+flight", "off"):
                    x = d[f"{tiles}__{center}__rot90__top1_err"].to_numpy() <= metric_m
                    y = d[f"{base}__{center}__rot90__top1_err"].to_numpy() <= metric_m
                    cells.append(f"{100 * x.mean():.1f}")
                    if tiles == base:
                        cells.append("—")
                    else:
                        wa, wb, pv = mcnemar(x, y)
                        cells.append(f"{100 * (x.mean() - y.mean()):+.1f} ({wa}/{wb}, {pv:.2f})")
                lines.append(f"| {model} | r{region} | {tiles} | {b['tile_m']:.0f} | "
                             f"{b['oracle_best_tile_median_m']:.1f} m | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def tables(models: list[str]) -> str:
    lines = []
    for model in models:
        evals = load_evals(model)
        for center in ("off", "map+flight"):
            lines.append(f"\n### {model}, centering {center}\n")
            lines.append("| Region / map | tile m | crop f (median) | " +
                         " | ".join(POLICY_LABEL) + " |")
            lines.append("|---|---:|---:|" + "---:|" * len(POLICY_LABEL))
            for region, e in evals.items():
                for tiles, b in e["tiles"].items():
                    res = b["centers"].get(center, {})
                    cells = []
                    for p in POLICY_LABEL:
                        m = res.get(p)
                        cells.append("—" if m is None else f"{100 * m['top1@100']:.1f}")
                    lines.append(f"| r{region} {tiles} | {b['tile_m']:.0f} | "
                                 f"{b['agl_crop_fraction_median']:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_agl()
    slope = None
    if (SCALE / "calibration_megaloc.csv").exists():
        slope = fig_calibration()
        fig_curves()
    models = [m for m in ("megaloc", "denseuav-vit") if load_evals(m)]
    for model in models:
        for center in ("off", "map+flight"):
            fig_results(model, center)
        fig_sensitivity(model, "map+flight")
    for region, tiles in (("11", "r11_t250"), ("06", "r06_t200"), ("05", "r05_t250"), ("10", "r10_t200")):
        fig_error_vs_agl("megaloc", region, tiles)
    text = "".join(headline(m, c) for m in models for c in ("map+flight", "off"))
    text += "".join(gated(m, c) for m in models for c in ("map+flight", "off"))
    text += aligned(models)
    text += tables(models) + "\n" + paired_tests(models)
    if slope is not None:
        text = f"calibration log-log slope: {slope:.3f}\n" + text
    (SCALE / "tables.md").write_text(text)
    print(text)
    print(f"figures in {FIGS}")


if __name__ == "__main__":
    main()
