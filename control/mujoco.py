"""SO-101 actuators and named cameras. The environment advances simulation time."""
import mujoco
import numpy as np
from scipy.optimize import least_squares

from contracts import FrameObservation, JOINT_NAMES, JointPositions


class MujocoArm:
    def __init__(self, model, data, prefix="", renderer=None, *, wrist_camera="wrist_cam"):
        self.model, self.data = model, data
        self.wrist_camera = prefix + wrist_camera
        self.qpos_ids = [model.joint(prefix + name).qposadr.item() for name in JOINT_NAMES]
        self.actuator_ids = [model.actuator(prefix + name).id for name in JOINT_NAMES]
        self.limits = model.actuator_ctrlrange[self.actuator_ids]
        self.pose_site = model.site(prefix + "gripperframe").id
        self.hand = model.body(prefix + "gripper").id
        self.base = model.body(prefix + "base").id
        self.ik_data = mujoco.MjData(model)
        self.renderer = renderer if renderer is not None else mujoco.Renderer(model, height=480, width=640)

    def read_positions(self):
        values = self.data.qpos[self.qpos_ids].copy()
        values[-1] = np.interp(values[-1], self.limits[-1], [0, 1])
        return JointPositions(tuple(values))

    def actuator_targets(self, target):
        values = np.array(target.values)
        values[-1] = np.interp(values[-1], [0, 1], self.limits[-1])
        if np.any(values < self.limits[:, 0]) or np.any(values > self.limits[:, 1]):
            raise ValueError("Joint target outside SO-101 actuator limits")
        return values

    def command(self, target: JointPositions):
        self.data.ctrl[self.actuator_ids] = self.actuator_targets(target)

    def commanded_positions(self):
        values = self.data.ctrl[self.actuator_ids].copy()
        values[-1] = np.interp(values[-1], self.limits[-1], [0, 1])
        return JointPositions(tuple(values))

    def render(self, cameras):
        images = {}
        for name, camera in cameras:
            self.renderer.update_scene(self.data, camera=camera)
            images[name] = self.renderer.render()
        return images

    def local_tcp(self):
        rotation = self.data.xmat[self.base].reshape(3, 3)
        return rotation.T @ (self.data.site_xpos[self.pose_site] - self.data.xpos[self.base])

    def observe(self, frame_no, prompt):
        images = self.render((("wrist", self.wrist_camera), ("front", "front")))
        return FrameObservation(frame_no, images, self.read_positions(), prompt, tuple(self.local_tcp() * 1000))

    def solve_top_down(self, xyz, opening):
        """Keep fingers downward, with the jaws closing along the base Y axis."""
        rotation = self.data.xmat[self.base].reshape(3, 3)
        target = self.data.xpos[self.base] + rotation @ np.asarray(xyz, dtype=float)
        upright = rotation @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        # Only the scratch state is used during IK; this never moves a live body.
        scratch = self.ik_data
        arm_ids = self.qpos_ids[:5]

        def residual(angles):
            scratch.qpos[arm_ids] = angles
            mujoco.mj_kinematics(self.model, scratch)
            rotation = scratch.xmat[self.hand].reshape(3, 3)
            return np.r_[scratch.site_xpos[self.pose_site] - target,
                         (0.06 * (rotation - upright)).ravel()]

        lower, upper = self.limits[:5].T
        start = np.clip(self.data.qpos[arm_ids], lower, upper)
        solution = least_squares(residual, start, bounds=(lower, upper),
                                 ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if np.linalg.norm(solution.fun[:3]) > 0.001 or np.linalg.norm(solution.fun[3:]) > 0.004:
            raise ValueError(f"Unreachable top-down SO-101 target {target.tolist()}")
        return JointPositions((*solution.x, opening))

    def close(self):
        self.renderer.close()


class DualArm:
    """Left then right for recording; each Cartesian interface uses its own base."""
    def __init__(self, model, data, *, wrist_camera="wrist_cam"):
        left = MujocoArm(model, data, "left_", wrist_camera=wrist_camera)
        self.arms = {"left": left, "right": MujocoArm(model, data, "right_", left.renderer, wrist_camera=wrist_camera)}

    def commanded_positions(self):
        return JointPositions(tuple(x for arm in self.arms.values() for x in arm.commanded_positions().values))

    def command(self, target):
        if len(target.values) != 12:
            raise ValueError("Sort requires both sets of six joint targets")
        for i, arm in enumerate(self.arms.values()):
            arm.command(JointPositions(target.values[6*i:6*i+6]))

    def observe(self, frame_no, prompt):
        cameras = [(side + "_wrist", arm.wrist_camera) for side, arm in self.arms.items()]
        images = self.arms["left"].render([*cameras, ("front", "front")])
        state = JointPositions(tuple(x for arm in self.arms.values() for x in arm.read_positions().values))
        poses = {side: tuple((arm.local_tcp() * 1000).tolist()) for side, arm in self.arms.items()}
        return FrameObservation(frame_no, images, state, prompt, poses)

    def close(self):
        self.arms["left"].close()
