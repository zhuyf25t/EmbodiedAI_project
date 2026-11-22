"""VISTA arm primitives, executed on a shared 10 Hz clock."""
import math
from copy import deepcopy
import numpy as np

from contracts import JointPositions
from control.skills import HOME, interpolate, set_gripper
from control.stability import compare_images


def primitive(description, axes=()):
    props = {axis: {"type": "number", "minimum": -100, "maximum": 100} for axis in axes}
    return {"description": description, "parameters": {"type": "object", "properties": props,
            "required": list(axes), "additionalProperties": False}}


ACTIONS = {
    "move_by": primitive("相对当前实测末端移动；基座轴，单位 mm；保持夹爪原目标。",
                         ("dx_mm", "dy_mm", "dz_mm")),
    "open_gripper": primitive("保持臂目标，张爪到位。"),
    "close_gripper": primitive("保持臂目标，闭爪；回执仅报告是否夹稳；抬升后看图确认。"),
    "observe": primitive("保持上一关节指令，直到连续两拍到位且各相机画面稳定。"),
}


def action_schemas(dual):
    actions = deepcopy(ACTIONS)
    actions["move_by"]["parameters"]["properties"]["grasp_yaw_deg"] = {
        "type": "number", "minimum": -45, "maximum": 45,
        "description": "夹爪朝向绝对角，0 沿本机 +Y 闭合，正角绕 +Z。省略保持原朝向。"}
    if not dual:
        return actions
    for name in ("move_by", "open_gripper", "close_gripper"):
        schema = actions[name]["parameters"]
        schema["properties"]["arm"] = {"type": "string", "enum": ["left", "right"]}
        schema["required"].append("arm")
    return actions


class ArmExecutor:
    def __init__(self, env, obs, tick):
        self.env, self.obs, self.tick = env, obs, tick
        self.target = env.arm.commanded_positions()
        self.info = {}
        self.pending_hold = None
        self.yaw = {side: 0.0 for side in env.arms}

    def home(self):
        """Return empty arms sequentially. Every interpolated and hold tick uses advance()."""
        begin = self.obs.frame_no
        for i, side in enumerate(self.env.arms):
            start = self.target
            index = i * 6
            desired = JointPositions(start.values[:index] + HOME.values + start.values[index+6:])
            if not self.env.home_path_clear(side, HOME):
                raise RuntimeError(f"Obstructed home path for {side}")
            distance = np.max(np.abs(np.subtract(desired.values, start.values)))
            for target in interpolate(start, desired, max(2, math.ceil(1.5 * distance / .065))):
                self.advance(target)
            if self._settle("home", desired, None, side):
                raise RuntimeError(f"Home did not settle for {side}")
            self.yaw[side] = 0.0
        return (begin, self.obs.frame_no)

    def advance(self, target):
        previous = self.obs
        self.obs, _, _, _, self.info = self.env.step(target)
        self.target = target
        self.tick(previous, target, self.obs, self.info)
        return previous

    def _pinched(self, side):
        if "pinch" in self.info:
            return self.info["pinch"][side]
        return self.info["fixed_contact"] and self.info["moving_contact"]

    def _tcp(self, side):
        return np.asarray(self.obs.tcp_mm[side] if self.env.dual else self.obs.tcp_mm)

    def _reached(self, goal, xyz, side):
        state = np.array(self.obs.state.values).reshape(-1, 6)
        desired = np.array(goal.values).reshape(-1, 6)
        if np.max(np.abs(state[:, :5] - desired[:, :5])) >= .03:
            return False
        if xyz is not None and np.linalg.norm(self._tcp(side) - xyz) > 3:
            return False
        return all(abs(s[-1] - g[-1]) <= .02 or (g[-1] == 0 and self._pinched(arm))
                   for arm, s, g in zip(self.env.arms, state, desired))

    def execute(self, name, args):
        """Hold the final command until two consecutive post-arrival frames agree."""
        if self.pending_hold is not None:
            pending_name, xyz, side = self.pending_hold
            if name != "observe" and not (pending_name == name and name in ("close_gripper", "open_gripper")
                                          and args.get("arm", "right") == side):
                return "HOLD_REQUIRED"
            return self._settle(pending_name, self.target, xyz, side)
        side = args.get("arm", "right")
        arm = self.env.arms[side]
        index = list(self.env.arms).index(side) * 6
        start = self.target
        arm_start = JointPositions(start.values[index:index+6])
        arm_goal = arm_start
        xyz = None
        ticks = 0
        if name == "move_by":
            delta = [args[k] for k in ("dx_mm", "dy_mm", "dz_mm")]
            xyz = self._tcp(side) + delta
            try:
                yaw = args.get("grasp_yaw_deg", self.yaw[side])
                arm_goal = arm.solve_top_down(xyz / 1000, arm_start.values[-1], yaw)
                self.yaw[side] = yaw
            except ValueError:
                return "UNREACHABLE"
            distance = max(abs(a - b) for a, b in zip(arm_goal.values[:5], arm_start.values[:5]))
            ticks = max(1, math.ceil(1.5 * distance / .13))
        elif name in ("open_gripper", "close_gripper"):
            arm_goal = set_gripper(arm_start, float(name == "open_gripper"))
            ticks = math.ceil(6 * abs(arm_goal.values[-1] - arm_start.values[-1]))
        elif name != "observe":
            raise ValueError(f"Unknown action {name}")

        goal = JointPositions(start.values[:index] + arm_goal.values + start.values[index+6:])
        for target in interpolate(start, goal, ticks):
            self.advance(target)
            if self.env.done:
                return "EPISODE_LIMIT"

        self.pending_hold = (name, xyz, side)
        return self._settle(name, goal, xyz, side)

    def _settle(self, name, goal, xyz, side):
        previous_ready = previous_contact = False
        for _ in range(20):
            previous = self.advance(goal)
            ready = self._reached(goal, xyz, side)
            contact = self._pinched(side)
            if previous_ready and ready:
                change = np.abs(np.subtract(self.obs.state.values, previous.state.values))
                change = change.reshape(-1, 6)
                quiet = change[:, :5].max() < .003 and change[:, 5].max() < .005
                if quiet and all(compare_images(previous.images[c], self.obs.images[c]) for c in self.obs.images):
                    self.pending_hold = None
                    if name == "close_gripper" and not (previous_contact and contact):
                        return "GRASP_NOT_CONFIRMED"
                    return None
            if self.env.done:
                return "EPISODE_LIMIT"
            previous_ready, previous_contact = ready, contact
        delta = np.abs(np.subtract(self.obs.state.values, previous.state.values)).reshape(-1,6)
        blocked_still = (delta[:,:5].max() < .003 and delta[:,5].max() < .005 and
                              all(compare_images(previous.images[c], self.obs.images[c]) for c in self.obs.images))
        if name == "move_by" and blocked_still and not ready:
            # A stationary physical stop is a failed move. Start the next plan at
            # measured joints, keeping the gripper's commanded force on any payload.
            start = list(self.target.values)
            index = list(self.env.arms).index(side) * 6
            start[index:index+5] = self.obs.state.values[index:index+5]
            self.target = JointPositions(tuple(start))
            self.pending_hold = None
            return "MOTION_BLOCKED"
        return "SETTLE_TIMEOUT"
