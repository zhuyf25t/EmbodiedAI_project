"""Run a paired positive and negative Lift control; save synchronized evidence."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import mkdtemp

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from contracts import CONTROL_HZ
from control.skills import move_to, set_gripper
from scene.env import TabletopEnv
from scene.lift import Lift


HOLD_Z = .085  # World metres above the tabletop, for both TCP trajectories.


def actions(arm, start, xy):
    target = start
    for phase, z, ticks in (("approach", .090, 16), ("descend", .017, 16)):
        for target in move_to(arm, target, (*xy, z), ticks):
            yield phase, target
    target = set_gripper(target, 0.0)
    for _ in range(12):
        yield "close", target
    for target in move_to(arm, target, (*xy, HOLD_Z), 20):
        yield "lift", target
    for _ in range(10):
        yield "hold", target


def video_frame(obs, row, name, font):
    front = obs.images["front"]
    w, h = front.shape[:2][::-1]
    canvas = Image.new("RGB", (w, h + 80), "#182028")
    canvas.paste(Image.fromarray(front), (0, 0))
    draw = ImageDraw.Draw(canvas)
    contact = f'{int(row["fixed_contact"])}/{int(row["moving_contact"])}'
    clearance_cm = row["cube_clearance_m"] * 100
    draw.text((14, h + 8), f'{name.upper()}   {row["phase"]}   {row["time_s"]:.1f} s'
              f'     Lift success: {int(row["success"])}', font=font, fill="white")
    draw.text((14, h + 42), f'Cube bottom {clearance_cm:.2f} cm   |   goal 5.00 cm'
              f'   |   jaw contacts {contact}', font=font, fill="#b9d4e7")
    return np.asarray(canvas)


def run_case(env, seed, name, output):
    obs, info = env.reset(seed=seed, inspect_hold=True)
    cube = env.task.cube
    initial_pose = env.data.qpos[env.task.free:env.task.free + 7].copy()
    target_xy = initial_pose[:2].copy()
    if name == "empty":
        target_xy[1] += .065

    rows = []
    font = ImageFont.load_default(size=19)
    writer = imageio_ffmpeg.write_frames(str(output / f"{name}.mp4"), (640, 560),
                                         fps=CONTROL_HZ, codec="libx264", macro_block_size=1)
    writer.send(None)

    def record(phase, observation, diagnostics):
        pos = env.data.geom_xpos[cube]
        tcp = env.data.site("gripperframe").xpos
        row = dict(tick=observation.frame_no, time_s=observation.frame_no / CONTROL_HZ,
                   phase=phase, tcp_z_m=float(tcp[2]),
                   cube_x_m=float(pos[0]), cube_y_m=float(pos[1]), cube_center_z_m=float(pos[2]),
                   **diagnostics)
        rows.append(row)
        writer.send(video_frame(observation, row, name, font))

    try:
        record("reset", obs, info)
        for phase, action in actions(env.arm, obs.state, target_xy):
            obs, _, terminated, truncated, info = env.step(action)
            record(phase, obs, info)
            if terminated or truncated:
                raise RuntimeError("Diagnostic trajectory ended before the hold")
    finally:
        writer.close()
    with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)

    hold = [r for r in rows if r["phase"] == "hold"]
    first = next((r for r in rows if r["success"]), None)
    both_jaws = any(r["fixed_contact"] and r["moving_contact"] for r in rows if r["phase"] == "close")
    return dict(initial_cube_pose=initial_pose.tolist(), ticks=obs.frame_no,
                initial_clearance_m=rows[0]["cube_clearance_m"],
                final_clearance_m=rows[-1]["cube_clearance_m"],
                max_clearance_m=max(r["cube_clearance_m"] for r in rows),
                final_tcp_z_m=rows[-1]["tcp_z_m"],
                first_success_s=first["time_s"] if first else None,
                first_success_clearance_m=first["cube_clearance_m"] if first else None,
                both_jaws_during_close=both_jaws,
                any_jaw_contact=any(r["fixed_contact"] or r["moving_contact"] for r in rows),
                hold_success_ticks=sum(r["success"] for r in hold), hold_ticks=len(hold),
                min_hold_clearance_m=min(r["cube_clearance_m"] for r in hold))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path("output/check_lift")
    root.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix=f"seed_{args.seed}_", dir=root))
    with TabletopEnv(Lift) as env:
        grasp = run_case(env, args.seed, "grasp", output)
        empty = run_case(env, args.seed, "empty", output)
    checks = dict(
        same_reset=grasp["initial_cube_pose"] == empty["initial_cube_pose"],
        starts_on_table=all(abs(c["initial_clearance_m"]) < .001 for c in (grasp, empty)),
        pinched_before_lifting=grasp["both_jaws_during_close"],
        grasp_held_above_threshold=grasp["hold_success_ticks"] == grasp["hold_ticks"],
        empty_no_success=empty["first_success_s"] is None,
        empty_no_contact=not empty["any_jaw_contact"],
        empty_stays_on_table=empty["max_clearance_m"] < .001,
        reached_commanded_height=all(abs(c["final_tcp_z_m"] - HOLD_Z) < .003 for c in (grasp, empty)),
        matching_final_tcp=abs(grasp["final_tcp_z_m"] - empty["final_tcp_z_m"]) < .002,
    )
    result = dict(seed=args.seed, mujoco_version=mujoco.__version__, control_hz=CONTROL_HZ,
                  threshold_m=Lift.height_threshold, grasp=grasp, empty=empty, checks=checks,
                  passed=all(checks.values()))
    report = json.dumps(result, indent=2, allow_nan=False)
    (output / "results.json").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Evidence: {output.resolve()}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
