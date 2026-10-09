"""Visual-inertial navigation core for GPS-denied AVL trajectory mode.

Pure-numpy building blocks (no torch / faiss / Qt):

* :mod:`avl.nav.frames`   - ENU tangent plane, quaternion algebra, gravity
* :mod:`avl.nav.imu_sim`  - synthetic IMU from a ground-truth trajectory
* :mod:`avl.nav.ins`      - strapdown mechanization
* :mod:`avl.nav.eskf`     - 15-state error-state Kalman filter
* :mod:`avl.nav.filters`  - filter registry + run loop
* :mod:`avl.nav.metrics`  - trajectory-error metrics

See docs/VisualInertial_AVL_Design.md.
"""

from avl.nav.eskf import ErrorStateEKF, EskfConfig, default_p0
from avl.nav.filters import FILTERS, Fix, FusionResult, run_filter
from avl.nav.frames import LocalFrame
from avl.nav.imu_sim import PRESETS, GradePreset, ImuStream, simulate_imu
from avl.nav.ins import NominalState, StateTrajectory, mechanize
from avl.nav.metrics import trajectory_metrics
from avl.nav.raster import RegionRaster, contiguous_run_along_path, snap_path_to_frames

__all__ = [
    "RegionRaster",
    "contiguous_run_along_path",
    "snap_path_to_frames",
    "ErrorStateEKF",
    "EskfConfig",
    "default_p0",
    "FILTERS",
    "Fix",
    "FusionResult",
    "run_filter",
    "LocalFrame",
    "PRESETS",
    "GradePreset",
    "ImuStream",
    "simulate_imu",
    "NominalState",
    "StateTrajectory",
    "mechanize",
    "trajectory_metrics",
]
