"""Shared seeded layouts and contact measurements for Stack and Sort."""
import mujoco
import numpy as np


class TwoCubes:
    wrist_camera = "wrist_wide"
    colors = ("red", "green")
    dual = False
    max_steps = 300
    stable_ticks = 5
    spawn_low = (.193, -.036)
    spawn_high = (.237, .036)
    min_separation = .046

    def __init__(self, model):
        self.cubes = {c: model.geom(c + "_cube").id for c in self.colors}
        self.joints = {c: model.joint(c + "_cube_free") for c in self.colors}
        self.table = model.geom("table").id
        self.jaws = {}
        for arm in (("left", "right") if self.dual else ("right",)):
            prefix = arm + "_" if self.dual else ""
            for i in range(model.ngeom):
                name = model.geom(i).name or ""
                if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                    name = model.mesh(model.geom_dataid[i]).name
                if not name.startswith(prefix):
                    continue
                name = name[len(prefix):]
                if name.startswith(("fixed_jaw_", "wrist_roll_follower_so101_gripper_part")):
                    self.jaws[i] = (arm, "fixed")
                elif name.startswith(("moving_jaw_box", "moving_jaw_sph", "moving_jaw_so101_gripper_part")):
                    self.jaws[i] = (arm, "moving")

    def reset(self, model, data, rng):
        while True:
            red, green = rng.uniform(self.spawn_low, self.spawn_high, (2, 2))
            if np.max(np.abs(red - green)) >= self.min_separation:
                break
        for color, xy in zip(self.colors, (red, green)):
            adr = self.joints[color].qposadr.item()
            data.qpos[adr:adr + 2] = xy
        self.streak = 0
        self.last_time = None

    def measure(self, model, data):
        pairs = set()
        jaws = {arm: {color: set() for color in self.colors}
                for arm in (("left", "right") if self.dual else ("right",))}
        force = np.zeros(6)
        for i, contact in enumerate(data.contact):
            a, b = map(int, contact.geom)
            if a not in self.cubes.values() and b not in self.cubes.values():
                continue
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] <= 0:
                continue
            pairs.add(frozenset((a, b)))
            for color, cube in self.cubes.items():
                if cube in (a, b):
                    other = b if a == cube else a
                    if other in self.jaws:
                        arm, side = self.jaws[other]
                        jaws[arm][color].add(side)
        pinch = {arm: any(len(sides) == 2 for sides in cubes.values()) for arm, cubes in jaws.items()}
        released = not any(sides for cubes in jaws.values() for sides in cubes.values())
        quiet = True
        for j in self.joints.values():
            adr = j.dofadr.item()
            velocity = data.qvel[adr:adr + 6]
            quiet &= np.linalg.norm(velocity[:3]) <= .012 and np.linalg.norm(velocity[3:]) <= .35
        arrangement, diagnostics = self.arrangement(model, data, pairs)
        if data.time != self.last_time:
            self.streak = self.streak + 1 if arrangement and released and quiet else 0
            self.last_time = data.time
        return dict(**diagnostics, pinch=pinch, released=released, quiet=bool(quiet),
                    stable_ticks=self.streak, success=self.streak >= self.stable_ticks)
