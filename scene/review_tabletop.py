"""Export the actual VISTA camera stream, play receipts and HDF5 alignment audit."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import h5py
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from collection.recorder import save_json
from contracts import CONTROL_HZ, DUAL_JOINT_NAMES, JOINT_NAMES
from scene.task_video import TaskVideo


def records(path):
    return [json.loads(line) for line in path.read_text("utf8").splitlines()]


def review(run, entry, manifest):
    ep = f"ep_{entry['seed']:04d}"
    task = manifest["task"]
    raw = run / "raw" / ep
    out = run.parents[2] / "output/vista" / task / run.name / ep
    events = records(out / "calls.jsonl")
    ticks = records(raw / "ticks.jsonl")
    cameras = manifest["cameras"]
    requests = {v["tool_call_id"]: v for v in events if v["event"] == "tool_request"}
    receipts = []
    labels = {0: "reset"}
    ends = {0, len(ticks)}
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
    save_json(out / "verification.json", dict(task=task, seed=entry["seed"], success=entry["success"],
              ticks=len(ticks), video_frames=len(judges), cameras=cameras, verified_hdf5=entry.get("hdf5"),
              play_rounds=len(receipts), final_judge=judges[-1]))
    print(out, entry["success"], len(ticks), judges[-1], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text("utf8"))
    for entry in manifest["episodes"]:
        if entry["status"] in ("finished", "interrupted"):
            review(args.run, entry, manifest)
