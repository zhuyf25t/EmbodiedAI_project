"""Provisional tabletop layout; distances are metres in the right-arm base frame."""
from pathlib import Path

import mujoco
import numpy as np


class Lift:
    prompt = "用右臂抓住红块，将其抬离桌面至少 5 cm。"
    model_file = Path(__file__).parent / "assets" / "so101" / "lift.xml"
    max_steps = 120
    height_threshold = 0.05
    # Provisional uniform reset area, aligned cube, until course coordinates arrive.
    spawn_low = (0.205, -0.010)
    spawn_high = (0.218, 0.010)

    def __init__(self, model):
        self.cube = model.geom("red_cube").id
        self.table = model.geom("table").id
        self.free = model.joint("red_cube_free").qposadr.item()
        # Only finger pads, tips and their collision meshes can certify a pinch.
        self.fixed_geoms = set()
        self.moving_geoms = set()
        for idx in range(model.ngeom):
            geom = model.geom(idx)
            name = geom.name or ""
            mesh_id = model.geom_dataid[idx]
            if model.geom_type[idx] == mujoco.mjtGeom.mjGEOM_MESH:
                name = model.mesh(mesh_id).name
            if name.startswith(("fixed_jaw_", "wrist_roll_follower_so101_gripper_part")):
                self.fixed_geoms.add(idx)
            if name.startswith(("moving_jaw_box", "moving_jaw_sph", "moving_jaw_so101_gripper_part")):
                self.moving_geoms.add(idx)

    def reset(self, model, data, rng):
        data.qpos[self.free:self.free + 2] = rng.uniform(self.spawn_low, self.spawn_high)

    def measure(self, model, data):
        # A rotated box's vertical half-extent is the sum of its projected half-axes.
        z_axis = data.geom_xmat[self.cube].reshape(3, 3)[2]
        bottom = data.geom_xpos[self.cube, 2] - np.abs(z_axis) @ model.geom_size[self.cube]
        table_top = data.geom_xpos[self.table, 2] + model.geom_size[self.table, 2]
        clearance = float(bottom - table_top)
        touching = set()
        wrench = np.zeros(6)
        for contact_index, contact in enumerate(data.contact):
            if self.cube not in contact.geom:
                continue
            other = contact.geom2 if contact.geom1 == self.cube else contact.geom1
            mujoco.mj_contactForce(model, data, contact_index, wrench)
            if wrench[0] > 0:
                touching.add(other)
        fixed = bool(self.fixed_geoms & touching)
        moving = bool(self.moving_geoms & touching)
        return dict(cube_clearance_m=clearance, fixed_contact=fixed, moving_contact=moving,
                    success=bool(fixed and moving and clearance >= self.height_threshold))
