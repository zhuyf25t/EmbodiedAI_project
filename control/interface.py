from typing import Protocol

from contracts import FrameObservation, JointPositions


class ArmControl(Protocol):
    def observe(self, frame_no: int, prompt: str) -> FrameObservation: ...

    def command(self, target: JointPositions) -> None: ...

    def solve_top_down(self, xyz: tuple[float, float, float], opening: float, yaw_deg: float = 0.0) -> JointPositions:
        """Solve a vertical tool pose in base coordinates (metres)."""
        ...
