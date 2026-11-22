"""Gym-style tabletop environment with one physics clock and a task-owned judge."""
import mujoco
import numpy as np

from contracts import CAMERAS, DUAL_CAMERAS, DUAL_JOINT_NAMES, JOINT_NAMES, CONTROL_HZ, JointPositions
from control.mujoco import DualArm, MujocoArm


class TabletopEnv:
    def __init__(self, task_type):
        path = task_type.model_file
        # A virtual filesystem also works with non-ASCII checkout paths on Windows.
        assets = {p.name: p.read_bytes() for p in (path.parent / "assets").glob("*.stl")}
        assets["so101.xml"] = (path.parent / "so101.xml").read_bytes()
        self.model = mujoco.MjModel.from_xml_string(path.read_text(encoding="utf-8"), assets=assets)
        self.data = mujoco.MjData(self.model)
        self.task = task_type(self.model)
        self.dual = getattr(task_type, "dual", False)
        adapter = DualArm if self.dual else MujocoArm
        self.arm = adapter(self.model, self.data, wrist_camera=getattr(task_type, "wrist_camera", "wrist_cam"))
        self.arms = self.arm.arms if self.dual else {"right": self.arm}
        self.camera_names = DUAL_CAMERAS if self.dual else CAMERAS
        self.joint_names = DUAL_JOINT_NAMES if self.dual else JOINT_NAMES
        self.substeps = round(1 / (CONTROL_HZ * self.model.opt.timestep))
        self.done = True

    def reset(self, *, seed=None, inspect_hold=False, finish_on_success=True, limit_ticks=True):
        """Collection checks success at settled action boundaries; evaluation can stop immediately."""
        mujoco.mj_resetData(self.model, self.data)
        # Top-down TCP near (170, 12, 78) mm. The wrist camera housing clears the arm.
        home = JointPositions((-.0014, -.5422, .5084, 1.6046, 1.6181, .8))
        for arm in self.arms.values():
            arm.command(home)
            self.data.qpos[arm.qpos_ids] = arm.actuator_targets(home)
        self.task.reset(self.model, self.data, np.random.default_rng(seed))
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=self.substeps * 3)
        mujoco.mj_forward(self.model, self.data)
        self.frame_no = 0
        self.done = False
        self.finish_on_success = finish_on_success and not inspect_hold
        self.tick_limit = self.task.max_steps if limit_ticks else None
        return self.arm.observe(0, self.task.prompt), self.task.measure(self.model, self.data)

    def step(self, target: JointPositions):
        reward, terminated, truncated, info = self.step_physics(target)
        obs = self.arm.observe(self.frame_no, self.task.prompt)
        return obs, reward, terminated, truncated, info

    def step_physics(self, target: JointPositions):
        """Advance the same control tick without rendering an unused observation."""
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished episode")
        self.arm.command(target)
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        mujoco.mj_forward(self.model, self.data)
        self.frame_no += 1
        info = self.task.measure(self.model, self.data)
        terminated = info["success"] and self.finish_on_success
        truncated = self.tick_limit is not None and self.frame_no >= self.tick_limit and not terminated
        self.done = terminated or truncated
        return float(info["success"]), terminated, truncated, info

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.arm.close()
