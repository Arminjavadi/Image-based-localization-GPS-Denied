"""What each method did on UAV-VisLoc — the progress report's findings as data.

The benchmark console shows these next to the recipe switches and on its Findings
tab, so the reason a step is on (and what it costs) is visible where it is chosen.
Numbers are top-1 within 100 m, 144 frames per region, measured in this repository
(docs/AVL_Progress_Report, 2 October 2026) unless marked otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Finding:
    key: str
    title: str
    #: adopted | gated | accuracy | not adopted | prototype
    status: str
    #: headline effect, short enough for a badge
    badge: str
    #: one line for the recipe panel
    summary: str
    #: where the idea comes from
    idea: str
    #: what this repository does
    did: str
    #: (setting, result) pairs, measured
    numbers: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    cost: str = ""
    needs: str = ""
    #: report section
    section: str = ""


FINDINGS: tuple[Finding, ...] = (
    Finding(
        key="ensemble",
        title="Encoder ensemble",
        status="accuracy",
        badge="+21 pts r05",
        summary="Average the cosine scores of encoders trained on different data.",
        idea="Multi-Process Fusion (Hausler 2019); Waheed et al. 2021 — methods that fail on "
        "different images combine best.",
        did="L2-normalise each encoder's descriptor and concatenate; the cosine is the mean of the "
        "members' cosines. Nothing is trained and every member has an equal vote.",
        numbers=(
            ("r05 best single (Game4Loc)", "59.7 %"),
            ("MegaLoc + Game4Loc + AnyLoc-L", "80.6 %"),
            ("… + heading", "85.4 %  ·  91.7 % fused"),
            ("r10 best pair (MegaLoc + AnyLoc-L)", "47.2 %  (MegaLoc alone 36.8 %)"),
        ),
        cost="sum of the members — 173 GMAC, 3.8× MegaLoc",
        needs="nothing",
        section="5.5",
    ),
    Finding(
        key="heading",
        title="Heading alignment",
        status="adopted",
        badge="+11 pts r05",
        summary="Turn the frame north-up from the compass; one encode instead of four.",
        idea="AnyVisLoc: a 60° heading error costs 25.7 points; FoundLoc keeps frames yaw-aligned; "
        "Kinnari et al. rotate frames with the drone's attitude.",
        did="Rotate each frame to the 90° step nearest north (exact angle where the altitude crop "
        "crops anyway) and encode it once.",
        numbers=(
            ("MegaLoc r05", "56.9 → 68.1 %"),
            ("AnyLoc-L r05", "49.3 → 66.0 %"),
            ("Ensemble r05", "80.6 → 85.4 %"),
            ("Game4Loc r05", "59.7 → 58.3 %  (no gain)"),
        ),
        cost="¼ of the compute — 1 encode instead of 4",
        needs="a heading per frame (compass / IMU); the flight CSV's yaw_deg",
        section="5.2",
    ),
    Finding(
        key="center",
        title="Domain centering",
        status="adopted",
        badge="+6 to +12 pts",
        summary="Subtract the map's and the flight's mean descriptor before scoring.",
        idea="Suzuki et al. 2013: centering removes hub vectors; test-time reference sets (2025).",
        did="Subtract the map's mean from the tiles once, and the running mean of the frames seen so "
        "far from each new frame (causal: past frames only).",
        numbers=(
            ("MegaLoc r05 (heading + flight mean)", "68.1 → 74.3 %"),
            ("MegaLoc r10", "16.0 → 27.8 %  ·  36.8 % with heading"),
            ("MegaLoc r06", "34.7 → 37.5 %  ·  43.1 % with heading"),
            ("Map mean only, Game4Loc r10", "16.7 → 5.6 %  (why flight mode)"),
        ),
        cost="one vector subtraction",
        needs="nothing (the first 10 frames lean on the map mean)",
        section="5.3",
    ),
    Finding(
        key="agl",
        title="Altitude scale",
        status="gated",
        badge="+39 pts r11",
        summary="Crop the frame to a tile's ground size when it covers > 1.25× a tile.",
        idea="Altitude-adaptive localization (2026) +41.5 pts; SUES-200: accuracy depends on height; "
        "Copernicus GLO-30 terrain.",
        did="Height above ground = barometric altitude − terrain (DEM). A frame covers k × AGL metres; "
        "it is centre-cropped to the tile's ground size only above the 1.25× gate.",
        numbers=(
            ("r11 MegaLoc (frame ≈ 1.9× tile)", "26.4 → 65.3 %  (58 won, 4 lost)"),
            ("Mean of r05 / r10 / r06 / r11", "45.1 → 55.4 %"),
            ("r05, r10, r06 (frames already fit)", "≈ no change"),
            ("±10 % altitude error", "no change; ±20 % keeps > 80 % of the gain"),
        ),
        cost="none — still one encode",
        needs="altitude above sea level, DEM tiles, the camera constant k",
        section="5.7",
    ),
    Finding(
        key="window",
        title="IMU search window",
        status="adopted",
        badge="+12–19 pts",
        summary="Search only tiles within 3σ of the navigation prior.",
        idea="STHN searches 512 m around the last fix; NGPS predicts the search region from "
        "visual-inertial odometry; FoundLoc keeps a sliding window.",
        did="In the benchmark the prior is simulated: the true position plus Gaussian noise of σ "
        "(seeded, identical for every encoder). In flight σ comes from the Kalman covariance.",
        numbers=(
            ("AnyLoc-L r05, σ 100 m", "49.3 → 61.8 %"),
            ("DenseUAV-ViT r06, σ 100 m", "27.8 → 46.5 %"),
            ("DenseUAV-ViT r10, σ 100 m", "18.8 → 37.5 %"),
            ("σ 50 m", "prior alone 86 % — confirm the IMU, don't replace it"),
        ),
        cost="cheaper search (21 of 384 tiles at σ 100 m)",
        needs="a navigation prior and its uncertainty",
        section="5.4",
    ),
    Finding(
        key="fusion",
        title="Top-5 fusion",
        status="adopted",
        badge="+6 pts r05",
        summary="Keep the largest geo-consistent group of the top-5 and average it.",
        idea="FoundLoc clusters the retrieved positions (DBSCAN) and keeps the largest cluster.",
        did="Weight the top-5 by score, group them within 150 m, average the heaviest group.",
        numbers=(
            ("Ensemble + heading r05", "85.4 → 91.7 %"),
            ("AnyLoc-L, window σ 100 m, r05", "61.8 → 70.1 %"),
            ("DenseUAV-ViT, window σ 100 m, r10", "37.5 → 49.3 %"),
            ("MegaLoc + altitude crop r11", "63.9 → 67.4 %"),
        ),
        cost="negligible",
        needs="nothing",
        section="5.6",
    ),
    Finding(
        key="rerank",
        title="Re-ranking, classic features",
        status="not adopted",
        badge="0 / 144 verified",
        summary="SIFT / ORB / AKAZE + RANSAC cannot bridge drone ↔ satellite.",
        idea="Pipelines verify the top tiles geometrically; AnyVisLoc: RoMa 70 % within 5 m, SIFT "
        "18.5 % on satellite maps.",
        did="Local features + RANSAC homography on the top-10, with a plausibility guard against "
        "degenerate fits.",
        numbers=(
            ("r10, 10 candidates", "0 verified on 144 / 144 frames"),
            ("Correct tile 45 m away", "~40 matches, every homography degenerate"),
            ("DenseUAV before the guard", "0.60 → 0.15  (false verifications)"),
        ),
        cost="+420–490 ms per frame",
        needs="a learned matcher (LightGlue, GIM, MatchAnything) — the next experiment",
        section="5.8",
    ),
    Finding(
        key="trajectory",
        title="Visual-inertial filter",
        status="prototype",
        badge="p95 652 → 308 m",
        summary="15-state error-state Kalman filter fusing fixes with a simulated IMU.",
        idea="FoundLoc (EKF on a Jetson Xavier NX); NGPS (UKF, 2.94 m RMSE); Solà's ESKF.",
        did="Visual fixes enter as position measurements behind a χ² gate that rejects outliers.",
        numbers=(
            ("r05, AnyLoc-lite, 50 frames", "median 70 → 68 m, worst 810 → 403 m"),
            ("χ² gate", "rejected 7 of the 9 fixes > 200 m wrong"),
            ("Simulated 5 km route", "IMU alone drifts 82 %; fused 59 m median"),
        ),
        cost="negligible",
        needs="a real IMU log (tested with a simulated one only); see the Trajectory tab",
        section="5.9",
    ),
)

FINDINGS_BY_KEY = {finding.key: finding for finding in FINDINGS}

#: Encoder facts from the report's table (§4): GMAC per image, parameters, and raw
#: top-1 within 100 m on r05 / r10 / r06 (no heading, no centering, 4 rotations).
#: None = not measured.
ENCODER_FACTS: dict[str, dict] = {
    "denseuav-vit": {"gmac": 4.2, "params": "22 M", "r05": 32.6, "r10": 18.8, "r06": 27.8},
    "game4loc": {"gmac": 49.3, "params": "86 M", "r05": 59.7, "r10": 16.7, "r06": 22.9},
    "infogeo-dense": {"gmac": 91.7, "params": "90.5 M", "r05": 72.2, "r10": 20.1, "r06": 27.1},
    "infogeo": {"gmac": 91.7, "params": "90.5 M", "r05": 63.9, "r10": None, "r06": None},
    "megaloc": {"gmac": 45.9, "params": "≈ 90 M", "r05": 56.9, "r10": 16.0, "r06": 34.7},
    "anyloc-lite": {"gmac": 21.9, "params": "86 M", "r05": 44.4, "r10": 18.1, "r06": 13.2},
    "anyloc-l": {"gmac": 77.8, "params": "304 M", "r05": 49.3, "r10": 28.5, "r06": None},
    "anyloc-sat": {"gmac": 82.0, "params": "300 M", "r05": 28.5, "r10": 4.9, "r06": None},
    "dinov3-b": {"gmac": 37.0, "params": "86 M", "r05": 47.2, "r10": None, "r06": None},
    "dinov2": {"gmac": 21.9, "params": "86 M", "r05": None, "r10": None, "r06": None},
    "mixvpr": {"gmac": 8.4, "params": "10.9 M", "r05": None, "r10": None, "r06": None},
    "cosplace": {"gmac": 17.5, "params": "46.7 M", "r05": None, "r10": None, "r06": None},
    "eigenplaces": {"gmac": 9.1, "params": "≈ 25 M", "r05": None, "r10": None, "r06": None},
    "sample4geo": {"gmac": 45.0, "params": "≈ 88 M", "r05": None, "r10": None, "r06": None},
}

GMAC = {name: facts["gmac"] for name, facts in ENCODER_FACTS.items()}

STATUS_LABEL = {
    "adopted": "adopted",
    "gated": "adopted · gated",
    "accuracy": "adopted for accuracy",
    "not adopted": "not adopted",
    "prototype": "prototype",
}
