"""A spawned worker owns one MuJoCo environment and never loads a network."""
import traceback

import numpy as np

from contracts import CAMERAS, JointPositions
from scene.env import TabletopEnv
from scene.lift import Lift
from .images import resize_rgb


def policy_input(obs, image_hw):
    rgb = np.stack([resize_rgb(obs.images[name], image_hw) for name in CAMERAS])
    return rgb.transpose(0, 3, 1, 2), np.asarray(obs.state.values, np.float32)


def run_worker(pipe, image_hw):
    try:
        with TabletopEnv(Lift) as env:
            limits = env.arm.limits.copy()
            limits[-1] = (0, 1)
            while True:
                op, payload = pipe.recv()
                if op == "close":
                    break
                if op == "reset":
                    seed = payload
                    obs, _ = env.reset(seed=seed)
                    trace = []
                    reply = (policy_input(obs, image_hw), None)
                elif op == "advance":
                    for predicted in payload:
                        if not np.isfinite(predicted).all():
                            raise FloatingPointError("Nonfinite ACT target")
                        target = JointPositions(tuple(np.clip(predicted, *limits.T)))
                        _, hit, timeout, judge = env.step_physics(target)
                        trace.append(dict(tick=env.frame_no, action=target.values, **judge))
                        if hit or timeout:
                            break
                    if env.done:
                        result = dict(seed=seed, success=hit, ticks=env.frame_no,
                                      reason="success" if hit else "timeout",
                                      peak_clearance_mm=1000 * max(t["cube_clearance_m"] for t in trace),
                                      ever_bilateral=any(t["fixed_contact"] and t["moving_contact"] for t in trace),
                                      trace=trace)
                        reply = (None, result)
                    else:
                        obs = env.arm.observe(env.frame_no, env.task.prompt)
                        reply = (policy_input(obs, image_hw), None)
                else:
                    raise ValueError(f"Unknown worker command {op}")
                pipe.send(("ok", reply))
    except EOFError:
        pass
    except Exception:
        pipe.send(("error", traceback.format_exc()))
    finally:
        pipe.close()
