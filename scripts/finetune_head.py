"""Train a retrieval head on cached frozen-DINOv2 features (CPU is enough).

Reads the caches written by scripts/finetune_features.py. Training regions and the
validation region come from ``train_rNN.npz``; test sets from ``eval_<name>.npz``.
The checkpoint is selected on the validation region only; the test sets are
evaluated once, at the end, next to the untrained backbone as a reference.

Loss: symmetric InfoNCE between drone frames and satellite crops. A frame's
similarity to a crop is the max over its four orientations, mirroring the
four-orientation search at test time. Each positive crop's ground scale is drawn at
random, so the head has to tolerate the unknown frame footprint. Half the batches
are spatial neighbourhoods (the negatives a search window leaves), half are random.
Crops centred within ``--fn-radius`` of a frame are masked out rather than treated
as negatives, since neighbouring frames overlap.

Example
-------
python scripts/finetune_head.py --train 01 02 03 04 07 08 09 --val 11 \
    --test r05_t250 r06_t200 r10_t200 --kind mix --tag mix_v1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from avl.finetune.head import RetrievalHead
from avl.geo import _from_local_xy, weighted_geo_fusion

FEATS = Path("artifacts/finetune/feats")
OUT = Path("artifacts/finetune")
PRIOR_SIGMAS = [50, 100, 200, 400]


def local_xy(lat, lon, lat0, lon0):
    la = math.radians(lat0)
    m_lat = 111_132.92 - 559.82 * math.cos(2 * la) + 1.175 * math.cos(4 * la)
    m_lon = 111_412.84 * math.cos(la) - 93.5 * math.cos(3 * la)
    return np.stack([(np.asarray(lon) - lon0) * m_lon, (np.asarray(lat) - lat0) * m_lat], -1)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class Region:
    def __init__(self, rid: str) -> None:
        d = np.load(FEATS / f"train_r{rid}.npz")
        self.rid = rid
        self.q = torch.from_numpy(d["q_feat"])          # (N, 4, 65, 768) f16
        self.t = torch.from_numpy(d["t_feat"])          # (N, S, 65, 768)
        self.g = torch.from_numpy(d["g_feat"])          # (G, 65, 768)
        self.scales = d["scales"]
        lat0, lon0 = float(d["lat"].mean()), float(d["lon"].mean())
        self.xy = local_xy(d["lat"], d["lon"], lat0, lon0)
        self.gxy = local_xy(d["g_lat"], d["g_lon"], lat0, lon0) if len(d["g_lat"]) else np.zeros((0, 2))
        self.lat, self.lon = d["lat"], d["lon"]
        self.g_lat, self.g_lon = d["g_lat"], d["g_lon"]

    def __len__(self) -> int:
        return len(self.q)


def eval_set_from_region(r: Region, scale: float = 250.0) -> dict:
    """Validation proxy: a region's GT-centred crops at one scale plus its random crops."""
    s = int(np.argmin(np.abs(r.scales - scale)))
    return {
        "r_feat": torch.cat([r.t[:, s], r.g]),
        "r_lat": np.concatenate([r.lat, r.g_lat]), "r_lon": np.concatenate([r.lon, r.g_lon]),
        "q_feat": r.q, "q_lat": r.lat, "q_lon": r.lon,
    }


def load_eval_set(name: str) -> dict:
    d = np.load(FEATS / f"eval_{name}.npz")
    return {
        "r_feat": torch.from_numpy(d["r_feat"]), "r_lat": d["r_lat"], "r_lon": d["r_lon"],
        "q_feat": torch.from_numpy(d["q_feat"]), "q_lat": d["q_lat"], "q_lon": d["q_lon"],
    }


# --------------------------------------------------------------------------- #
# evaluation (mirrors scripts/visloc_eval.py, incl. its seeded prior draws)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def describe(fn, feats: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    flat = feats.reshape(-1, *feats.shape[-2:])
    out = torch.cat([fn(flat[i : i + chunk]) for i in range(0, len(flat), chunk)])
    return out.reshape(*feats.shape[:-2], -1)


def raw_descriptor(tokens: torch.Tensor) -> torch.Tensor:
    """Untrained reference: mean of the patch grid (as avl.models.dinov2 does)."""
    return F.normalize(tokens[:, 1:].float().mean(1), dim=-1)


def retrieval_metrics(desc_q, desc_r, es: dict, prior_sigma=None, prior_k=3.0, seed=0, top_k=5) -> dict:
    scores = torch.einsum("qrd,nd->qrn", desc_q, desc_r).amax(1).numpy()  # (Q, N)
    Q = len(scores)
    q_lat, q_lon, r_lat, r_lon = es["q_lat"], es["q_lon"], es["r_lat"], es["r_lon"]
    offsets = np.random.default_rng(seed).normal(0.0, prior_sigma or 1.0, size=(Q, 2))
    top1, best5, fused = np.zeros(Q), np.zeros(Q), np.zeros(Q)
    for i in range(Q):
        xy = local_xy(r_lat, r_lon, q_lat[i], q_lon[i])
        dist = np.hypot(xy[:, 0], xy[:, 1])
        s = scores[i].copy()
        if prior_sigma:
            c_lat, c_lon = _from_local_xy(offsets[i, 0], offsets[i, 1], q_lat[i], q_lon[i])
            cxy = local_xy(r_lat, r_lon, c_lat, c_lon)
            cd = np.hypot(cxy[:, 0], cxy[:, 1])
            win = cd <= prior_k * prior_sigma
            if win.sum() < top_k:
                win[np.argsort(cd)[:top_k]] = True
            s[~win] = -np.inf
        idx = np.argsort(-s)[:top_k]
        top1[i], best5[i] = dist[idx[0]], dist[idx].min()
        pose = weighted_geo_fusion(r_lat[idx], r_lon[idx], s[idx], method="cluster")
        pxy = local_xy(pose.latitude, pose.longitude, q_lat[i], q_lon[i])
        fused[i] = float(np.hypot(*pxy))
    return {
        "top1_100": float((top1 <= 100).mean()), "top5_100": float((best5 <= 100).mean()),
        "fused_100": float((fused <= 100).mean()), "fused_50": float((fused <= 50).mean()),
        "top1_median_m": float(np.median(top1)), "fused_median_m": float(np.median(fused)),
        "fused_p95_m": float(np.percentile(fused, 95)),
    }


def full_report(fn, es: dict) -> dict:
    dq, dr = describe(fn, es["q_feat"]), describe(fn, es["r_feat"])
    rep = {"global": retrieval_metrics(dq, dr, es)}
    for sig in PRIOR_SIGMAS:
        rep[f"prior{sig}"] = retrieval_metrics(dq, dr, es, prior_sigma=sig)
    return rep


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def make_batch(r: Region, rng, batch: int, local: bool) -> np.ndarray:
    n = len(r)
    if n <= batch:
        return rng.permutation(n)
    if local:
        anchor = rng.integers(n)
        d = np.hypot(*(r.xy - r.xy[anchor]).T)
        return np.argsort(d)[:batch]
    return rng.choice(n, batch, replace=False)


def train_step(head, r: Region, idx, rng, args):
    B = len(idx)
    q = head(r.q[idx])                                           # (B, 4, D)
    s_idx = rng.integers(r.t.shape[1], size=B)
    t = head(r.t[idx, s_idx])                                    # (B, D)
    txy = r.xy[idx]
    if len(r.g) and args.gallery_per_batch:
        g_idx = rng.choice(len(r.g), min(args.gallery_per_batch, len(r.g)), replace=False)
        t = torch.cat([t, head(r.g[g_idx])])
        txy = np.concatenate([txy, r.gxy[g_idx]])
    sim = torch.einsum("brd,td->brt", q, t).amax(1)               # (B, B+G)
    dist = np.hypot(*(r.xy[idx][:, None, :] - txy[None, :, :]).transpose(2, 0, 1))
    mask = torch.from_numpy(dist < args.fn_radius)
    mask[torch.arange(B), torch.arange(B)] = False
    logits = (sim / args.tau).masked_fill(mask, float("-inf"))
    target = torch.arange(B)
    loss_q = F.cross_entropy(logits, target)
    loss_t = F.cross_entropy(logits[:, :B].T, target)
    return 0.5 * (loss_q + loss_t)


def selection_score(rep: dict) -> float:
    return 0.5 * (rep["global"]["top1_100"] + rep["prior100"]["top1_100"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--val", nargs="*", default=[])
    p.add_argument("--test", nargs="*", default=[])
    p.add_argument("--kind", choices=["gem", "mix"], default="gem")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--out-dim", type=int, default=512)
    p.add_argument("--mix-width", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--gallery-per-batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--fn-radius", type=float, default=100.0)
    p.add_argument("--local-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="head")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    train = [Region(r) for r in args.train]
    val = {f"val_r{r}": eval_set_from_region(Region(r)) for r in args.val}
    print(f"[{args.tag}] train {[(r.rid, len(r)) for r in train]}  val {list(val)}", flush=True)
    weights = np.array([len(r) for r in train], dtype=float)
    weights /= weights.sum()
    steps = max(1, int(sum(len(r) for r in train) / args.batch))

    head = RetrievalHead(args.kind, args.hidden, args.out_dim, args.mix_width, args.dropout)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps, pct_start=0.1)

    ckpt = OUT / f"head_{args.tag}.pt"
    history, best = [], (-1.0, -1)
    if val:
        base = {k: full_report(raw_descriptor, es) for k, es in val.items()}
        print(f"[{args.tag}] untrained val: " + "  ".join(
            f"{k} top1<100 {v['global']['top1_100']:.3f}/{v['prior100']['top1_100']:.3f}" for k, v in base.items()),
            flush=True)
    for epoch in range(args.epochs):
        head.train()
        t0, losses = time.perf_counter(), []
        for _ in range(steps):
            r = train[rng.choice(len(train), p=weights)]
            idx = make_batch(r, rng, args.batch, rng.random() < args.local_frac)
            loss = train_step(head, r, idx, rng, args)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        head.eval()
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "s": time.perf_counter() - t0}
        if val:
            row["val"] = {k: full_report(head, es) for k, es in val.items()}
            score = float(np.mean([selection_score(v) for v in row["val"].values()]))
            row["score"] = score
            if score > best[0]:
                best = (score, epoch + 1)
                head.save(ckpt, args=vars(args), epoch=epoch + 1, val=row["val"])
            desc = "  ".join(f"{k} top1<100 {v['global']['top1_100']:.3f}/{v['prior100']['top1_100']:.3f}"
                             for k, v in row["val"].items())
        else:
            head.save(ckpt, args=vars(args), epoch=epoch + 1)
            desc = ""
        history.append(row)
        print(f"[{args.tag}] ep {epoch + 1:2d} loss {row['loss']:.3f} ({row['s']:.0f}s)  {desc}", flush=True)

    head = RetrievalHead.load(ckpt)
    print(f"[{args.tag}] best epoch {best[1] if val else args.epochs} -> {ckpt}", flush=True)
    report = {"args": vars(args), "best_epoch": best[1] if val else args.epochs, "history": history, "test": {}}
    for name in args.test:
        es = load_eval_set(name)
        report["test"][name] = {"untrained": full_report(raw_descriptor, es), "head": full_report(head, es)}
        for which in ("untrained", "head"):
            rr = report["test"][name][which]
            print(f"[{args.tag}] {name:9s} {which:9s} " + "  ".join(
                f"{k}: top1<100 {v['top1_100']:.3f} fused<100 {v['fused_100']:.3f} med {v['fused_median_m']:.0f}m"
                for k, v in rr.items() if k in ("global", "prior100", "prior200")), flush=True)
    (OUT / f"report_{args.tag}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
