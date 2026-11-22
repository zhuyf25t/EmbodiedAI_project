"""Raw control ticks are durable for every outcome; publish complete successes to HDF5."""
import json
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from contracts import CAMERAS, CONTROL_HZ, JOINT_NAMES, SCENE_VERSION


def save_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    # A Windows reader can briefly hold the manifest open without delete sharing.
    for attempt in range(40):
        try:
            temp.replace(path)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 39:
                raise
            time.sleep(0.05)


class EpisodeRecorder:
    def __init__(self, directory, storage, joint_names=JOINT_NAMES, cameras=CAMERAS):
        self.joint_names, self.cameras = joint_names, cameras
        self.raw = directory
        self.storage = storage
        self.tick_file = (directory / "ticks.jsonl").open("x", encoding="utf8", buffering=1)
        self.count = 0
        self.phase = "home"
        self.home_end_frame = 0

    def tick(self, before, target, after, diagnostics):
        if before.frame_no != self.count or after.frame_no != self.count + 1:
            raise RuntimeError("Discontinuous control frames")
        self.storage.ensure_frame(after)
        row = dict(phase=self.phase, frame=before.frame_id, next_frame=after.frame_id, state=before.state.values,
                   action=target.values, judge_after=diagnostics)
        self.tick_file.write(json.dumps(row, allow_nan=False) + "\n")
        self.count += 1

    def close(self):
        self.tick_file.close()

    def export(self, path, task, prompt, seed, source):
        rows = [json.loads(line) for line in (self.raw / "ticks.jsonl").read_text("utf8").splitlines()]
        if not rows or len(rows) != self.count or not rows[-1]["judge_after"]["success"]:
            raise ValueError("Incomplete successful trajectory")
        frame_dir = self.raw / "frames"
        temp = path.with_suffix(".partial")
        with h5py.File(temp, "x") as h:
            h.attrs.update(task=task, prompt=prompt, seed=seed, fps=CONTROL_HZ, joint_names=json.dumps(self.joint_names),
                           units=json.dumps((["rad"] * 5 + ["opening_0_1"]) * (len(self.joint_names) // 6)), source=source,
                           scene_version=SCENE_VERSION, home_end_frame=self.home_end_frame)
            h.create_dataset("action", data=np.asarray([r["action"] for r in rows], np.float32))
            h.create_dataset("observations/state", data=np.asarray([r["state"] for r in rows], np.float32))
            h.attrs["camera_names"] = json.dumps(self.cameras)
            images = {name: h.create_dataset(f"observations/images/{name}", (len(rows), 480, 640, 3),
                      dtype="u1", chunks=(1, 480, 640, 3), compression="lzf") for name in self.cameras}
            for i, row in enumerate(rows):
                if row["frame"] != f"f{i:06d}" or row["next_frame"] != f"f{i+1:06d}":
                    raise ValueError("Misaligned control tick")
                for camera, dataset in images.items():
                    with Image.open(frame_dir / camera / (row["frame"] + ".png")) as img:
                        dataset[i] = np.asarray(img)
        temp.replace(path)
