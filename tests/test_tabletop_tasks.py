"""Restore boundary cases independently of VISTA's perception or actions."""
import unittest

import mujoco
import numpy as np

from scene.env import TabletopEnv
from scene.sort import Sort
from scene.stack import Stack


def restore(env, poses):
    for color, (x, y, z, yaw) in poses.items():
        joint = env.task.joints[color]
        pos, vel = joint.qposadr.item(), joint.dofadr.item()
        env.data.qpos[pos:pos+7] = [x, y, z, np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        env.data.qvel[vel:vel+6] = 0
    mujoco.mj_forward(env.model, env.data)


class TabletopTaskTest(unittest.TestCase):
    def test_supported_stack_requires_centering_and_distinct_stable_ticks(self):
        with TabletopEnv(Stack) as env:
            for offset, expected in ((.010, False), (.003, True)):
                env.reset(seed=0, finish_on_success=False)
                restore(env, {"green": (.22, -.025, .015, 0), "red": (.22 + offset, -.025, .045, 0)})
                target = env.arm.commanded_positions()
                for tick in range(1, 8):
                    info = env.step_physics(target)[-1]
                    self.assertTrue(info["red_supported"])
                    self.assertEqual(info["success"], expected and tick >= 5)
                    self.assertEqual(env.task.measure(env.model, env.data), info)

    def test_sort_requires_the_entire_matching_footprint(self):
        with TabletopEnv(Sort) as env:
            for red, green, inside in (
                ((.17, -.06, .015, 0), (.27, .06, .015, 0), (True, True)),
                ((.27, .06, .015, 0), (.17, -.06, .015, 0), (False, False)),
                ((.182, -.06, .015, np.pi/4), (.27, .06, .015, 0), (False, True)),
            ):
                env.reset(seed=0, finish_on_success=False)
                restore(env, {"green": green, "red": red})
                for _ in range(6):
                    info = env.step_physics(env.arm.commanded_positions())[-1]
                self.assertEqual(info["inside"], dict(zip(("red", "green"), inside)))
                self.assertEqual(info["success"], all(inside))


if __name__ == "__main__":
    unittest.main()
