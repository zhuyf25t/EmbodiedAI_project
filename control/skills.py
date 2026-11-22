"""Generate joint targets. The caller clocks and records every step."""
import numpy as np

from contracts import JointPositions
from control.interface import ArmControl


HOME = JointPositions((-.0014, -.5422, .5084, 1.6046, 1.6181, .8))


def move_to(arm: ArmControl, start: JointPositions, xyz, ticks):
    """Interpolate in joint space to the IK solution for a top-down tool pose."""
    goal = arm.solve_top_down(xyz, start.values[-1])
    yield from interpolate(start, goal, ticks)


def interpolate(start: JointPositions, goal: JointPositions, ticks):
    initial = np.asarray(start.values)
    delta = np.asarray(goal.values) - initial
    for k in range(1, ticks + 1):
        t = k / ticks
        blend = t * t * (3 - 2 * t)
        yield JointPositions(tuple(initial + blend * delta))


def set_gripper(start: JointPositions, opening: float):
    return JointPositions((*start.values[:5], opening))
