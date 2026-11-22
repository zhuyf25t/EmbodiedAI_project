"""Track 1 measured cells, oriented cube sampling and collision-checked reset paths."""
import math

import mujoco
import numpy as np

from contracts import JointPositions
from control.skills import HOME

HALF = .015


def edge_yaw(degrees):
    """Nearest square edge to the neutral jaw direction, in degrees."""
    return (degrees + 45) % 90 - 45


def square_corners(xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(xy) + HALF * np.array([[-1,-1],[-1,1],[1,1],[1,-1]]) @ np.array([[c,s],[-s,c]])


def overlap(a, b):
    for corners in (a, b):
        for side in (corners[1]-corners[0], corners[2]-corners[1]):
            axis = side / np.linalg.norm(side)
            pa, pb = a @ axis, b @ axis
            if pa.max() + .001 < pb.min() or pb.max() + .001 < pa.min():
                return False
    return True


class Layout:
    def __init__(self, env):
        self.env = env
        self.model = env.model
        self.scratch = mujoco.MjData(self.model)
        self.roots = {self.model.body(side + "_base").id for side in ("left", "right")}
        self.table = self.model.geom("table").id
        self.cubes = set(env.task.cubes.values())

    def clear(self, arm, angles, *, avoid_cubes):
        data = self.scratch
        data.qpos[arm.qpos_ids] = angles
        mujoco.mj_forward(self.model, data)
        for contact in data.contact:
            if contact.dist >= -.0004:
                continue
            a, b = contact.geom
            ra, rb = self.model.body_rootid[self.model.geom_bodyid[[a,b]]]
            if arm.base not in (ra, rb):
                continue
            other = b if ra == arm.base else a
            if (ra in self.roots and rb in self.roots) or other == self.table or (avoid_cubes and other in self.cubes):
                return False
        return True

    def safe_segment(self, arm, start, end, avoid_cubes):
        count = max(2, math.ceil(np.max(np.abs(end-start))/.10)+1)
        return all(self.clear(arm, q, avoid_cubes=avoid_cubes) for q in np.linspace(start, end, count))

    def local(self, side, xy, z):
        arm = self.env.arms[side]
        return self.env.data.xmat[arm.base].reshape(3,3).T @ (np.r_[xy,z]-self.env.data.xpos[arm.base])

    def path(self, side, source, yaw, destination):
        arm = self.env.arms[side]
        high = self.env.task.carry_height
        poses = [(source, high), (source, .017), (source, high)]
        if destination is not None:
            poses += [(destination, high), (destination, .021 if self.env.dual else .051), (destination, high)]
        self.scratch.qpos[:] = self.env.data.qpos
        previous = arm.actuator_targets(HOME)
        try:
            for xy, z in poses:
                q = arm.actuator_targets(arm.solve_top_down(self.local(side, xy, z), .8, yaw))
                if not self.safe_segment(arm, previous, q, False):
                    return False
                previous = q
        except ValueError:
            return False
        return True

    def arrange(self, rng):
        env = self.env
        site = env.data.site("mixed" if env.dual else "red_target")
        half = self.model.site_size[site.id,:2]
        for attempt in range(1000):
            poses = {}
            for color in env.task.colors:
                angle = rng.uniform(-np.pi, np.pi)
                margin = HALF * (abs(np.sin(angle))+abs(np.cos(angle)))
                xy = site.xpos[:2] + rng.uniform(-half+margin, half-margin)
                poses[color] = (xy, angle)
            if len(poses)==2 and overlap(*(square_corners(*pose) for pose in poses.values())):
                continue
            red, theta = poses["red"]
            green = poses.get("green", (None, None))[0]
            dest = env.data.site("red_target").xpos[:2] if env.dual else green
            if not self.path("right", red, edge_yaw(np.degrees(theta)), dest):
                continue
            if env.dual and not self.path("left", green, edge_yaw(np.degrees(poses["green"][1])),
                                          env.data.site("green_target").xpos[:2]):
                continue
            for color,(xy,yaw) in poses.items():
                adr = env.task.joints[color].qposadr.item()
                env.data.qpos[adr:adr+7] = [*xy, HALF, np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
            return
        raise RuntimeError("Could not sample a reachable Track 1 arrangement")

    def random_arms(self, rng):
        env = self.env
        for _ in range(160):
            for arm in env.arms.values():
                # Modest, independent departures from a known overhead pose, with empty jaws.
                delta = rng.uniform(-1,1,6)*np.array([.36,.23,.25,.22,.34,.16])
                v = np.array(HOME.values) + delta
                v[:5] = np.clip(v[:5],arm.limits[:5,0]+.01,arm.limits[:5,1]-.01)
                kin = JointPositions(tuple(v))
                arm.command(kin)
                env.data.qpos[arm.qpos_ids] = arm.actuator_targets(kin)
            self.scratch.qpos[:] = env.data.qpos
            for arm in env.arms.values():
                if not self.safe_segment(arm, env.data.qpos[arm.qpos_ids], arm.actuator_targets(HOME), True):
                    break
            else:
                return
        raise RuntimeError("Could not sample a collision-free random home path")
