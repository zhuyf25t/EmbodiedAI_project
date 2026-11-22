"""Physics-only positive and negative controls for Stack and Sort, with optional live GUI."""
import argparse
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

from collection.recorder import save_json
from control.vista import ArmExecutor
from scene.env import TabletopEnv
from scene.sort import Sort
from scene.stack import Stack
from scene.task_video import Preview, TaskVideo


def check(env, seed, positive, out, preview):
    obs, initial = env.reset(seed=seed, finish_on_success=False, limit_ticks=False)
    expected = "positive" if positive else "negative"
    video = TaskVideo(out / (expected + ".mp4"), env.camera_names)
    poses = {c: env.data.geom_xpos[g].tolist() for c, g in env.task.cubes.items()}
    rows, calls = [], []
    phase = "reset"

    def record(obs, info):
        frame = video.add(obs.images, f"{expected} | {phase} | tick {obs.frame_no}", info)
        if preview:
            preview.show(frame)
        if obs.frame_no == 0:
            frame.save(out / (expected + "_initial.jpg"))
            if preview and expected == "positive":
                from PIL import ImageGrab
                import sys
                if sys.platform == "win32":
                    ImageGrab.grab(window=preview.root.winfo_id()).save(out / "gui_initial.png")
        rows.append(dict(frame=obs.frame_no, phase=phase, **info))
        return frame

    try:
        record(obs, initial)
        executor = ArmExecutor(env, obs, lambda before, target, after, info: record(after, info))

        def issue(side, name, args):
            nonlocal phase
            phase = f"{side} {name}"
            if env.dual and name != "observe":
                args["arm"] = side
            for attempt in range(3):
                error = executor.execute(name, args)
                calls.append(dict(side=side, action=name, args=args.copy(), end=executor.obs.frame_no, error=error))
                if error != "SETTLE_TIMEOUT":
                    break
                if name != "close_gripper":
                    name, args = "observe", {}
            if error:
                raise RuntimeError(f"{phase}: {error}")

        def goto(side, xy, z):
            arm = env.arms[side]
            target = env.data.xmat[arm.base].reshape(3, 3).T @ (np.r_[xy, z] - env.data.xpos[arm.base])
            delta = 1000 * (target - arm.local_tcp())
            issue(side, "move_by", dict(zip(("dx_mm", "dy_mm", "dz_mm"), delta.tolist())))

        assignments = [("right", "red")]
        if env.dual and positive:
            assignments.append(("left", "green"))
        for side, color in assignments:
            source = env.data.geom_xpos[env.task.cubes[color], :2].copy()
            if env.dual:
                destination = env.data.site(color + "_target").xpos[:2].copy()
            else:
                destination = env.data.geom_xpos[env.task.cubes["green"], :2].copy()
                if not positive:
                    destination[0] -= .045
            goto(side, source, .079)
            goto(side, source, .017)
            issue(side, "close_gripper", {})
            goto(side, source, .079)
            goto(side, destination, .079)
            goto(side, destination, .051 if positive and not env.dual else .021)
            issue(side, "open_gripper", {})
            goto(side, destination, .076)
        for _ in range(3):
            issue("right", "observe", {})
        result = rows[-1]["success"]
        if result != positive or (not positive and any(r["success"] for r in rows)):
            raise AssertionError(f"Unexpected terminal predicate for {expected}: {rows[-1]}")
        if env.dual and not positive and rows[-1]["inside"] != {"red": True, "green": False}:
            raise AssertionError("Sort negative control did not finish its first delivery")
        return poses
    finally:
        video.close()
        save_json(out / (expected + ".json"), dict(seed=seed, poses=poses, calls=calls, ticks=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stack","sort"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    base = Path("output") / ("check_" + args.task)
    base.mkdir(exist_ok=True, parents=True)
    out = Path(mkdtemp(dir=base, prefix=f"seed{args.seed}-"))
    preview = Preview(args.task.title() + " physics self-check") if args.gui else None
    try:
        with TabletopEnv({"stack": Stack, "sort": Sort}[args.task]) as env:
            good = check(env, args.seed, True, out, preview)
            bad = check(env, args.seed, False, out, preview)
            assert np.allclose(list(good.values()), list(bad.values()))
        print("Verified both controls:", out, flush=True)
    finally:
        if preview:
            preview.close()


if __name__ == "__main__":
    main()
