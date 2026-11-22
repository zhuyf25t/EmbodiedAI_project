"""Raw control ticks are durable for every outcome; publish complete successes to HDF5."""
import json
import os
from tempfile import NamedTemporaryFile
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


def export_demo(raw, path, task, prompt, entry, joint_names, cameras):
    """Publish once. A crash after publication can be reconciled against raw."""
    rows = [json.loads(line) for line in (raw / "ticks.jsonl").read_text("utf8").splitlines()]
    if not rows or len(rows) != entry["ticks"] or not rows[-1]["judge_after"]["success"]:
        raise ValueError("Incomplete successful trajectory")
    for i, row in enumerate(rows):
        if (row["frame"], row["next_frame"]) != (f"f{i:06d}", f"f{i+1:06d}"):
            raise ValueError("Misaligned control tick")
    attrs = dict(task=task, prompt=prompt, seed=entry["seed"], fps=CONTROL_HZ,
                 joint_names=json.dumps(joint_names), camera_names=json.dumps(cameras),
                 units=json.dumps((["rad"] * 5 + ["opening_0_1"]) * (len(joint_names) // 6)),
                 source="vista", scene_version=SCENE_VERSION, home_end_frame=entry["home_end_frame"])
    arrays = {field: np.asarray([r[col] for r in rows], np.float32)
              for field, col in (("action","action"), ("observations/state","state"))}
    existing = path.exists()
    if existing:
        target = path
    else:
        with NamedTemporaryFile(dir=path.parent, prefix=path.stem+"-", suffix=".partial", delete=False) as temp:
            target = Path(temp.name)
    try:
        with h5py.File(target, "r" if existing else "w") as h:
            if existing:
                if any(h.attrs.get(k) != v for k, v in attrs.items()):
                    raise ValueError(f"Existing HDF5 metadata differs: {path}")
            else:
                h.attrs.update(attrs)
            for field, values in arrays.items():
                if existing:
                    if not np.array_equal(h[field][:], values):
                        raise ValueError(f"Existing HDF5 {field} differs: {path}")
                else:
                    h.create_dataset(field, data=values)
            for camera in cameras:
                name = f"observations/images/{camera}"
                shape = (len(rows), 480, 640, 3)
                images = h[name] if existing else h.create_dataset(name, shape, dtype="u1",
                                  chunks=(1, 480, 640, 3), compression="lzf")
                if images.shape != shape or images.dtype != np.uint8:
                    raise ValueError(f"Invalid camera stream: {name}")
                for i, row in enumerate(rows):
                    with Image.open(raw / "frames" / camera / (row["frame"] + ".png")) as img:
                        if existing:
                            if not np.array_equal(images[i], np.asarray(img)):
                                raise ValueError(f"Existing HDF5 image differs: {camera}/{i}")
                        else:
                            images[i] = np.asarray(img)
        if not existing:
            os.link(target, path)  # Atomic create-if-absent on a local filesystem.
    finally:
        if not existing:
            target.unlink(missing_ok=True)
