"""One isolated MuJoCo environment records every evaluated control tick."""
import traceback

import cv2
import imageio_ffmpeg
import numpy as np

from contracts import JointPositions
from scene.env import TabletopEnv
from scene.tasks import TASKS
from scene.task_video import TaskVideo
from .files import atomic_json
from .images import resize_rgb


def policy_input(obs, cameras, image_hw):
    rgb = np.stack([resize_rgb(obs.images[name], image_hw) for name in cameras])
    return rgb.transpose(0, 3, 1, 2), np.asarray(obs.state.values, np.float32)


def run_episode(pipe, env, task, seed, image_hw, directory, limits):
    obs, judge = env.reset(seed=seed)
    trace = []
    partial = directory / f"seed-{seed}.partial.mp4"
    video = TaskVideo(partial, env.camera_names)
    try:
        try:
            video.add(obs.images, f"{task} | seed {seed} | tick 0", judge)
            while not env.done:
                pipe.send(("ok", (policy_input(obs, env.camera_names, image_hw), None)))
                command, targets = pipe.recv()
                if command == "close":
                    return False
                if command != "advance" or targets.ndim != 2 or targets.shape[0] == 0 or targets.shape[1] != len(limits):
                    raise ValueError("Expected a chunk of joint targets for the active task")
                for predicted in targets:
                    if not np.isfinite(predicted).all():
                        raise FloatingPointError("Non-finite ACT joint target")
                    target = JointPositions(tuple(np.clip(predicted, *limits.T)))
                    obs, _, hit, timeout, judge = env.step(target)
                    trace.append(dict(tick=obs.frame_no, action=target.values, **judge))
                    video.add(obs.images, f"{task} | seed {seed} | tick {obs.frame_no}", judge)
                    if hit or timeout:
                        break
        finally:
            video.close()
        frames, _ = imageio_ffmpeg.count_frames_and_secs(partial)
        if frames != env.frame_no + 1:
            raise RuntimeError(f"Seed {seed}: encoded {frames} frames for {env.frame_no} control ticks")
        name = f"seed-{seed}-{'success' if hit else 'failure'}.mp4"
        record = dict(seed=seed, success=hit, reason="success" if hit else "timeout",
                      ticks=env.frame_no, video=name, video_frames=frames, final_judge=judge)
        if task == "lift":
            record.update(peak_clearance_mm=1000 * max(t["cube_clearance_m"] for t in trace),
                          ever_bilateral=any(t["fixed_contact"] and t["moving_contact"] for t in trace))
        destination = directory / name
        partial.replace(destination)
        try:
            atomic_json(directory / f"seed-{seed}.json", dict(**record, trace=trace))
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        pipe.send(("ok", (None, record)))
        return True
    finally:
        partial.unlink(missing_ok=True)


def run_worker(pipe, task, image_hw, directory):
    try:
        cv2.setNumThreads(1)
        with TabletopEnv(TASKS[task]) as env:
            # Each arm exposes actuator angles, but ACT uses normalized gripper openings.
            limits = np.concatenate([arm.limits for arm in env.arms.values()])
            limits[5::6] = (0, 1)
            while True:
                command, seed = pipe.recv()
                if command == "close":
                    break
                if command != "reset":
                    raise ValueError(f"Expected reset, got {command}")
                if not run_episode(pipe, env, task, seed, image_hw, directory, limits):
                    break
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        pipe.send(("error", traceback.format_exc()))
    finally:
        pipe.close()
