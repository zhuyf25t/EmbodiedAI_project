"""SO-101 single-arm measurements and targets, in policy order."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CONTROL_HZ = 10
CAMERAS = ("wrist", "front")
DUAL_CAMERAS = ("left_wrist", "right_wrist", "front")
DUAL_JOINT_NAMES = tuple(f"{side}_{j}" for side in ("left", "right") for j in JOINT_NAMES)


@dataclass(frozen=True)
class JointPositions:
    """Five angles in radians, followed by normalized gripper opening."""
    values: tuple[float, ...]

    def __post_init__(self):
        v = np.asarray(self.values, dtype=float)
        if (v.shape not in ((6,), (12,)) or not np.isfinite(v).all()
                or np.any((v[5::6] < 0) | (v[5::6] > 1))):
            raise ValueError("Expected five finite angles and a gripper opening in [0, 1] for each arm")
        object.__setattr__(self, "values", tuple(v.tolist()))


@dataclass(frozen=True, eq=False)
class FrameObservation:
    frame_no: int
    images: Mapping[str, np.ndarray]
    state: JointPositions
    prompt: str
    tcp_mm: tuple[float, float, float] | dict[str, tuple] | None = None

    @property
    def frame_id(self):
        return f"f{self.frame_no:06d}"

    @property
    def image(self):
        """Front RGB for callers written against the original single-camera interface."""
        return self.images["front"]

    def __post_init__(self):
        # Same ownership and RGB contract as MuJoCo-VISTA-ACT observations.py.
        if type(self.frame_no) is not int or self.frame_no < 0:
            raise ValueError("Frame index must be a nonnegative integer")
        cameras = DUAL_CAMERAS if len(self.state.values) == 12 else CAMERAS
        if set(self.images) != set(cameras):
            raise ValueError(f"Expected synchronized images for {cameras}")
        snapshots = {}
        for camera in cameras:
            rgb = self.images[camera]
            if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or not rgb.size:
                raise ValueError(f"Expected nonempty uint8 RGB for {camera}")
            snapshots[camera] = np.frombuffer(rgb.tobytes(), np.uint8).reshape(rgb.shape)
        object.__setattr__(self, "images", MappingProxyType(snapshots))


@dataclass(frozen=True)
class ObservationWindow:
    frames: tuple[FrameObservation, ...]

    def __post_init__(self):
        if not self.frames or any(b.frame_no != a.frame_no + 1 for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError("Expected a contiguous nonempty observation window")

    @property
    def current(self):
        return self.frames[-1]
