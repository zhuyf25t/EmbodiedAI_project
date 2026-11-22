"""Physics-only positive and negative controls for all three tasks, with a live GUI and recordings."""
import argparse
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

from collection.recorder import save_json
from control.vista import ArmExecutor
from scene.env import TabletopEnv
from scene.sort import Sort
from scene.lift import Lift
from scene.layout import edge_yaw
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
        def tick(before, target, after, info):
            record(after, info)
            rows[-1].update(before=before.frame_no, state=before.state.values, action=target.values,
                            tcp_after=after.tcp_mm)
        executor = ArmExecutor(env, obs, tick)
        phase = "home"
        executor.home()

        def issue(side, name, args, expected_error=None):
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
            if error != expected_error:
                raise RuntimeError(f"{phase}: {error}")

        def goto(side, xy, z, yaw):
            arm = env.arms[side]
            target = env.data.xmat[arm.base].reshape(3, 3).T @ (np.r_[xy, z] - env.data.xpos[arm.base])
            current = arm.local_tcp().copy()
            count = max(1, int(np.ceil(np.max(np.abs(target-current))*1000 / 94)))
            for waypoint in np.linspace(current, target, count+1)[1:]:
                delta = 1000 * (waypoint - arm.local_tcp())
                issue(side, "move_by", dict(dx_mm=delta[0], dy_mm=delta[1], dz_mm=delta[2], grasp_yaw_deg=yaw))

        assignments = [("right", "red")]
        if env.dual and positive:
            assignments.append(("left", "green"))
        for side, color in assignments:
            source = env.data.geom_xpos[env.task.cubes[color], :2].copy()
            rot = env.data.geom_xmat[env.task.cubes[color]].reshape(3,3)
            yaw = float(edge_yaw(np.degrees(np.arctan2(rot[1,0], rot[0,0]))))
            high = env.task.carry_height
            if isinstance(env.task, Lift):
                if not positive:
                    sites = np.array([[.45,.205], [.52,.205], [.50,.285]])
                    source = max(sites, key=lambda xy: np.linalg.norm(xy-source))
                goto(side, source, high, yaw)
                goto(side, source, .017, yaw)
                issue(side, "close_gripper", {}, None if positive else "GRASP_NOT_CONFIRMED")
                goto(side, source, high, yaw)
                break
            if env.dual:
                destination = env.data.site(color + "_target").xpos[:2].copy()
            else:
                destination = env.data.geom_xpos[env.task.cubes["green"], :2].copy()
                if not positive:
                    destination = source.copy()
            goto(side, source, high, yaw)
            goto(side, source, .017, yaw)
            issue(side, "close_gripper", {})
            goto(side, source, high, yaw)
            goto(side, destination, high, yaw)
            goto(side, destination, .051 if positive and not env.dual else .021, yaw)
            issue(side, "open_gripper", {})
            goto(side, destination, high, yaw)
        for _ in range(3):
            issue("right", "observe", {})
        result = rows[-1]["success"]
        if result != positive or (not positive and any(r["success"] for r in rows)):
            raise AssertionError(f"Unexpected terminal predicate for {expected}: {rows[-1]}")
        if env.dual and not positive and rows[-1]["inside"] != {"red": True, "green": False}:
            raise AssertionError("Sort negative control did not finish its first delivery")
        if isinstance(env.task, Lift) and not positive:
            assert max(r['cube_clearance_m'] for r in rows) < .001
            assert not any(r['fixed_contact'] or r['moving_contact'] for r in rows)
        return poses
    finally:
        video.close()
        save_json(out / (expected + ".json"), dict(seed=seed, poses=poses, calls=calls, ticks=rows))


def main(default_task=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("lift","stack","sort"), default=default_task, required=default_task is None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    base = Path("output") / ("check_" + args.task)
    base.mkdir(exist_ok=True, parents=True)
    out = Path(mkdtemp(dir=base, prefix=f"seed{args.seed}-"))
    print(f"Evidence: {out}", flush=True)
    preview = Preview(args.task.title() + " physics self-check") if args.gui else None
    try:
        with TabletopEnv({"lift": Lift, "stack": Stack, "sort": Sort}[args.task]) as env:
            good = check(env, args.seed, True, out, preview)
            bad = check(env, args.seed, False, out, preview)
            assert np.allclose(list(good.values()), list(bad.values()))
        print("Verified both controls:", out, flush=True)
    finally:
        if preview:
            preview.close()


if __name__ == "__main__":
    main()
