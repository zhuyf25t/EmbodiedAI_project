"""Export the actual VISTA camera stream, play receipts and HDF5 alignment audit."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import h5py
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from collection.batch import attempt_path, read_manifest
from collection.recorder import save_json
from contracts import CONTROL_HZ, DUAL_JOINT_NAMES, JOINT_NAMES, JointPositions, SCENE_VERSION
from scene.task_video import TaskVideo


def records(path):
    return [json.loads(line) for line in path.read_text("utf8").splitlines()]


def replay_actions(task, entry, raw, ticks):
    """Reapply recorded targets from random f0; audit state, judges and camera boundaries."""
    from scene.env import TabletopEnv
    from scene.lift import Lift
    from scene.stack import Stack
    from scene.sort import Sort

    edges = sorted({0, max(0, entry.get("home_end_frame",0)-1), entry.get("home_end_frame",0), len(ticks)})
    image_errors = []
    with TabletopEnv({"lift":Lift,"stack":Stack,"sort":Sort}[task]) as env:
        obs, _ = env.reset(seed=entry["seed"],finish_on_success=False,limit_ticks=False)
        for i in range(len(ticks)+1):
            if i in edges:
                if i:
                    obs = env.arm.observe(i, env.task.prompt)
                for name, rgb in obs.images.items():
                    with Image.open(raw/"frames"/name/(obs.frame_id+".png")) as saved:
                        error = np.abs(rgb.astype("i2")-np.asarray(saved))
                        image_errors.append(float(error.mean()))
            if i == len(ticks):
                break
            row = ticks[i]
            state = [v for arm in env.arms.values() for v in arm.read_positions().values]
            np.testing.assert_allclose(state, row["state"], atol=1e-7, rtol=0)
            info = env.step_physics(JointPositions(tuple(row["action"])))[-1]
            for key, value in info.items():
                if isinstance(value, (float, np.floating)):
                    np.testing.assert_allclose(value, row["judge_after"][key], atol=1e-7, rtol=0)
                elif value != row["judge_after"][key]:
                    raise ValueError(f"Replay mismatch at {i+1}: {key}")
    return dict(ticks=len(ticks), frames_compared=edges, max_camera_mean_abs_error=max(image_errors))


def review(run, entry, manifest, replay=False, *, evidence=None):
    ep = f"ep_{entry['seed']:04d}"
    task = manifest["task"]
    if evidence is None:  # Historical, per-task run directory.
        raw = run / "raw" / ep
        out = run.parents[2] / "output/vista" / task / run.name / ep
    else:
        out, raw = evidence, evidence / "raw"
    events = records(out / "calls.jsonl")
    ticks = records(raw / "ticks.jsonl")
    cameras = manifest["cameras"]
    requests = {v["tool_call_id"]: v for v in events if v["event"] == "tool_request"}
    receipts = []
    labels = {0: "reset"}
    home = entry.get("home_end_frame", 0)
    for i, tick in enumerate(ticks):
        if "phase" in tick and entry["status"] == "finished":
            assert tick["phase"] == ("home" if i < home else "vista")
        if tick.get("phase") == "home":
            labels[i + 1] = "control home"
    ends = {0, max(0, home-1), home, len(ticks)}
    for event in events:
        if event["event"] != "tool_result":
            continue
        result = event["result"].get("data") or {}
        if "action_results" not in result:
            continue
        receipts.append(dict(play=result["batch_id"], request=requests[event["tool_call_id"]], receipt=event))
        for a in result["action_results"]:
            if a["frame_range"]:
                start, end = a["frame_range"]
                args = a["args"]
                axes = " ".join(f"{args[k]:+.1f}" for k in ("dx_mm","dy_mm","dz_mm") if k in args)
                label = f"play {result['batch_id']} | {args.get('arm','right')} {a['action']} {axes}"
                for i in range(start + 1, end + 1):
                    if i in labels:
                        raise ValueError("Overlapping action ticks")
                    labels[i] = label
                ends.add(end)
    if entry["status"] == "finished" and (len(receipts) != entry["play_rounds"] or len(labels) != len(ticks)+1):
        raise ValueError("Control ticks do not agree with play receipts")
    if home and receipts:
        assert receipts[0]["request"]["args"]["basis_frame_id"] == f"f{home:06d}"
        save_json(out / "home_handoff.json", dict(home_end_frame=home,
            first_play=receipts[0]["request"]["args"],
            ticks=[ticks[i] for i in sorted({0,home-2,home-1,home,home+1}) if 0 <= i < len(ticks)]))
    save_json(out / "play_receipts.json", receipts)
    judges = [json.loads((raw / "reset.json").read_text("utf8"))["judge"]] + [t["judge_after"] for t in ticks]
    if entry["status"] == "finished" and judges[-1]["success"] != entry["success"]:
        raise ValueError("Terminal judge disagrees with collection")
    video_path = out / "episode.mp4"
    video = TaskVideo(video_path, cameras)
    thumbs = []
    try:
        with ExitStack() as files:
            if entry.get("success") and not entry.get("hdf5"):
                raise ValueError("Successful demonstration is missing its HDF5")
            h = files.enter_context(h5py.File(run / entry["hdf5"], "r")) if entry.get("hdf5") else None
            if h is not None:
                names = DUAL_JOINT_NAMES if task == "sort" else JOINT_NAMES
                if "scene_version" in manifest:
                    assert h.attrs["scene_version"] == manifest["scene_version"] and h.attrs["home_end_frame"] == home
                assert tuple(json.loads(h.attrs["joint_names"])) == names
                assert json.loads(h.attrs["camera_names"]) == cameras and h.attrs["task"] == task
                assert h.attrs["seed"] == entry["seed"] and h.attrs["fps"] == CONTROL_HZ
                assert h["action"].shape == h["observations/state"].shape == (len(ticks), len(names))
                for key, values in (("action", "action"), ("observations/state", "state")):
                    assert np.array_equal(h[key][:], np.asarray([t[values] for t in ticks], np.float32))
            for i, judge in enumerate(judges):
                frame = f"f{i:06d}"
                if i < len(ticks):
                    assert ticks[i]["frame"] == frame and ticks[i]["next_frame"] == f"f{i+1:06d}"
                with ExitStack() as inputs:
                    images = {c: inputs.enter_context(Image.open(raw / "frames" / c / (frame + ".png"))) for c in cameras}
                    if h is not None and i < len(ticks):
                        for c, image in images.items():
                            assert np.array_equal(h["observations/images/" + c][i], np.asarray(image))
                    painted = video.add(images, f"{task} | seed {entry['seed']} | {frame} | {labels.get(i, 'interrupted')}", judge)
                    if i in ends:
                        thumb = painted.crop((0,0,painted.width,360))
                        thumb.thumbnail((len(cameras)*160,120))
                        thumbs.append((i, thumb))
    finally:
        video.close()
    width = len(cameras)*160
    sheet = Image.new("RGB", (width*2, 145*((len(thumbs)+1)//2)), "#182028")
    draw = ImageDraw.Draw(sheet)
    for index, (i, thumb) in enumerate(thumbs):
        x, y = index%2*width, index//2*145
        sheet.paste(thumb,(x,y))
        draw.text((x+4,y+122), f"{i} | {labels.get(i,'interrupted')}", fill="white")
    sheet.save(out / "action_frames.jpg", quality=90)
    reader = imageio_ffmpeg.read_frames(str(video_path))
    meta = next(reader)
    assert meta["fps"] == CONTROL_HZ and sum(1 for _ in reader) == len(judges)
    physical = None
    if replay:
        if manifest.get("scene_version") != SCENE_VERSION:
            raise ValueError("Replay requires the matching scene revision")
        physical = replay_actions(task,entry,raw,ticks)
    save_json(out / "verification.json", dict(physics_replay=physical, task=task, seed=entry["seed"], success=entry["success"],
              home_end_frame=home, ticks=len(ticks), video_frames=len(judges), cameras=cameras, verified_hdf5=entry.get("hdf5"),
              play_rounds=len(receipts), final_judge=judges[-1]))
    print(out, entry["success"], len(ticks), judges[-1], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--seed", type=int, help="Review one seed")
    parser.add_argument("--attempt", type=int, help="Review one attempt number for that seed")
    parser.add_argument("--replay", action="store_true", help="Replay recorded targets in the matching physics scene")
    args = parser.parse_args()
    if args.attempt is not None and (args.seed is None or args.attempt < 1):
        parser.error("--attempt requires --seed and a positive attempt number")
    run = args.run.resolve()
    if (run / "manifest.json").is_file():
        manifest = json.loads((run / "manifest.json").read_text("utf8"))
        if "episodes" not in manifest:
            parser.error("specify a task directory, e.g. data/run01/sort")
        for entry in manifest["episodes"]:
            if entry["status"] in ("finished", "interrupted") and (args.seed is None or args.seed == entry["seed"]):
                review(run, entry, manifest, replay=args.replay)
    else:
        batch = read_manifest(run.parent)
        task = batch["tasks"][run.name]
        metadata = dict(task=run.name, cameras=task["cameras"], scene_version=batch["scene_version"])
        reviewed = 0
        for row in task["episodes"].values():
            for entry in row:
                if ((args.seed is not None and args.seed != entry["seed"])
                        or (args.attempt is not None and args.attempt != entry["attempt"])
                        or entry["status"] not in ("finished", "interrupted")):
                    continue
                evidence = attempt_path(run.parent, run.name, entry["seed"], entry["attempt"])
                if not (evidence / "raw/reset.json").is_file() or not (evidence / "raw/ticks.jsonl").is_file():
                    print(f"No recording available: {evidence}")
                    continue
                review(run.parent, entry, metadata, args.replay, evidence=evidence)
                reviewed += 1
        if (args.seed is not None or args.attempt is not None) and not reviewed:
            parser.error("no recorded attempt matched the selection")
