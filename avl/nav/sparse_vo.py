"""Frame-to-frame visual odometry for a nadir camera (sparse frames, no IMU).

Between two consecutive frames A and B: SIFT + ratio test, then a RANSAC
similarity (rotation, uniform scale, shift - ``cv2.estimateAffinePartial2D``).
The centre of B mapped into A's pixels is the image-space displacement; the
ground sample distance from A's height above ground (``k * AGL / width``) turns
it into metres, and A's heading (compass direction of the image top, clockwise
from north - UAV-VisLoc's ``yaw_deg``) rotates it into east/north.

A similarity, not a homography: at 400-2500 m above ground the scene is close
to planar and the camera close to nadir, so the 4-DoF model is better
conditioned and its scale doubles as a sanity check against the altitude ratio.

Camera tilt: the image centre is not the point below the drone. With pitch/roll
of a few degrees it lands AGL * tan(tilt) away (65 m at 367 m AGL and 10 deg),
and that changes from frame to frame. The nadir pixel is used instead, offset by
``f * tan(angle)`` per axis. On UAV-VisLoc the axes are ``Omega -> +x (right)``
and ``Kappa -> +y (down)``. That was found by testing all eight axis/sign
assignments: it is the best on r05, r06 and r11 and cuts r05's median step error
from 48 % to 12 %. The dataset does not document it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VoStep:
    ok: bool
    east_m: float = float("nan")
    north_m: float = float("nan")
    inliers: int = 0
    matches: int = 0
    scale: float = float("nan")      # B pixels -> A pixels (>1: B flew lower)
    rot_deg: float = float("nan")    # image rotation B -> A
    reason: str = ""


class SparseVO:
    def __init__(self, long_side: int = 1200, ratio: float = 0.8, min_inliers: int = 25,
                 ransac_px: float = 4.0, max_features: int = 4000) -> None:
        self.long_side = long_side
        self.ratio = ratio
        self.min_inliers = min_inliers
        self.ransac_px = ransac_px
        self.sift = cv2.SIFT_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def features(self, gray: np.ndarray):
        return self.sift.detectAndCompute(gray, None)

    def step(self, feat_a, feat_b, size_a: tuple[int, int], size_b: tuple[int, int],
             gsd_a_m: float, heading_a_deg: float, scale_hint: float | None = None,
             nadir_a: tuple[float, float] | None = None,
             nadir_b: tuple[float, float] | None = None) -> VoStep:
        """Displacement A -> B in metres (east, north).

        ``size_*`` are (width, height) of the images the features came from and
        ``gsd_a_m`` is metres per pixel of A at that size. ``scale_hint`` (AGL_B /
        AGL_A from the barometer) rejects fits whose scale disagrees by > 25 %.
        ``nadir_*`` are the pixels below the drone (:func:`nadir_pixel`); default
        is the image centre.
        """
        (kp_a, des_a), (kp_b, des_b) = feat_a, feat_b
        if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
            return VoStep(False, reason="few features")
        pairs = self.matcher.knnMatch(des_b, des_a, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < self.ratio * n.distance]
        if len(good) < self.min_inliers:
            return VoStep(False, matches=len(good), reason="few matches")
        pb = np.float32([kp_b[m.queryIdx].pt for m in good])
        pa = np.float32([kp_a[m.trainIdx].pt for m in good])
        M, mask = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC,
                                              ransacReprojThreshold=self.ransac_px,
                                              maxIters=5000, confidence=0.999)
        if M is None:
            return VoStep(False, matches=len(good), reason="no fit")
        inl = int(mask.sum())
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        if inl < self.min_inliers:
            return VoStep(False, inliers=inl, matches=len(good), scale=scale, rot_deg=rot,
                          reason="few inliers")
        if scale_hint is not None and abs(np.log(scale * scale_hint)) > np.log(1.25):
            # B's pixels are AGL_B/AGL_A times A's, so scale should be ~ 1/scale_hint
            return VoStep(False, inliers=inl, matches=len(good), scale=scale, rot_deg=rot,
                          reason="scale disagrees with altitude")
        nb = nadir_b if nadir_b is not None else (size_b[0] / 2.0, size_b[1] / 2.0)
        ca = np.asarray(nadir_a if nadir_a is not None else (size_a[0] / 2.0, size_a[1] / 2.0))
        cb = np.array([nb[0], nb[1], 1.0])
        dx, dy = (M @ cb - ca) * gsd_a_m          # image axes: x right, y down
        forward, right = -dy, dx
        h = np.radians(heading_a_deg)
        east = forward * np.sin(h) + right * np.cos(h)
        north = forward * np.cos(h) - right * np.sin(h)
        return VoStep(True, float(east), float(north), inl, len(good), scale, rot)


def nadir_pixel(size: tuple[int, int], focal_px: float, omega_deg: float, kappa_deg: float) -> tuple[float, float]:
    """Pixel below the drone for UAV-VisLoc attitude (see the module docstring)."""
    w, h = size
    return (w / 2.0 + focal_px * np.tan(np.radians(omega_deg)),
            h / 2.0 + focal_px * np.tan(np.radians(kappa_deg)))
