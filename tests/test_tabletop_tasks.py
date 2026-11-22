"""Restore boundary cases independently of VISTA's perception or actions."""
import unittest

import mujoco
import numpy as np

from control.skills import HOME
from control.stability import compare_images
from control.vista import ArmExecutor
from scene.env import TabletopEnv
from scene.lift import Lift
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
    def test_stationary_missed_goal_can_be_replanned(self):
        with TabletopEnv(Lift) as env:
            obs, _ = env.reset(seed=5,finish_on_success=False,limit_ticks=False)
            executor = ArmExecutor(env,obs,lambda *args:None)
            executor.home()
            actual = np.array(executor.obs.tcp_mm)
            # Stationary measured pose differs from the pending Cartesian target.
            error = executor._settle("move_by",executor.target,actual+[40,0,0],"right")
            self.assertEqual(error,"MOTION_BLOCKED")
            self.assertIsNone(executor.pending_hold)
            self.assertEqual(executor.target.values[:5],executor.obs.state.values[:5])
            self.assertIsNone(executor.execute("observe",{}))

    def test_subpixel_edges_do_not_hide_motion(self):
        rgb = np.full((480,640,3), 180, np.uint8)
        rgb[200:240,300:330] = [190,30,40]
        self.assertTrue(compare_images(rgb,np.roll(rgb,1,axis=1)))
        self.assertFalse(compare_images(rgb,np.roll(rgb,3,axis=1)))

    def test_home_uses_one_continuous_action_stream_from_random_f0(self):
        with TabletopEnv(Lift) as env:
            start, _ = env.reset(seed=5)
            initial = np.array(start.state.values)
            self.assertGreater(np.max(np.abs(initial-HOME.values)), .03)
            ticks = []
            executor = ArmExecutor(env, start, lambda before,target,after,info:
                                   ticks.append((before.frame_no, target.values, after.frame_no)))
            first, end = executor.home()
            self.assertEqual((first,end),(0,len(ticks)))
            self.assertTrue(all((before, after) == (i,i+1) for i,(before,_,after) in enumerate(ticks)))
            self.assertEqual(ticks[-1][1], ticks[-2][1])
            self.assertEqual(ticks[-1][1], HOME.values)
            self.assertLess(np.max(np.abs(np.subtract(ticks[0][1],initial))), .065)
            again, _ = env.reset(seed=5)
            np.testing.assert_array_equal(again.state.values, initial)
            changed, _ = env.reset(seed=6)
            self.assertGreater(np.max(np.abs(np.subtract(changed.state.values,initial))), .03)

    def test_supported_stack_requires_centering_and_distinct_stable_ticks(self):
        with TabletopEnv(Stack) as env:
            for offset, expected in ((.010, False), (.003, True)):
                env.reset(seed=0, finish_on_success=False)
                restore(env, {"green": (.48, .25, .015, 0), "red": (.48 + offset, .25, .045, 0)})
                target = env.arm.commanded_positions()
                for tick in range(1, 8):
                    info = env.step_physics(target)[-1]
                    self.assertTrue(info["red_supported"])
                    self.assertEqual(info["success"], expected and tick >= 5)
                    self.assertEqual(env.task.measure(env.model, env.data), info)

    def test_sort_requires_the_entire_matching_footprint(self):
        with TabletopEnv(Sort) as env:
            for red, green, inside in (
                ((.479, .25, .015, 0), (.121, .25, .015, 0), (True, True)),
                ((.121, .25, .015, 0), (.479, .25, .015, 0), (False, False)),
                ((.545, .25, .015, np.pi/4), (.121, .25, .015, 0), (False, True)),
            ):
                env.reset(seed=0, finish_on_success=False)
                restore(env, {"green": green, "red": red})
                for _ in range(6):
                    info = env.step_physics(env.arm.commanded_positions())[-1]
                self.assertEqual(info["inside"], dict(zip(("red", "green"), inside)))
                self.assertEqual(info["success"], all(inside))


if __name__ == "__main__":
    unittest.main()
