"""Episode-level, immutable train/validation sets from a named VISTA batch."""
from bisect import bisect_right
import json
from pathlib import Path
import random

import h5py
import numpy as np
from torch.utils.data import Dataset

from contracts import CAMERAS, CONTROL_HZ, DUAL_CAMERAS, DUAL_JOINT_NAMES, JOINT_NAMES, SCENE_VERSION
from .files import file_digest, read_json
from .images import resize_rgb


def sensors(task):
    if task == "sort":
        return DUAL_CAMERAS, DUAL_JOINT_NAMES
    if task in ("lift", "stack"):
        return CAMERAS, JOINT_NAMES
    raise ValueError(f"Unknown task: {task}")


def snapshot(batch, task):
    manifest = read_json(batch / "manifest.json")
    cameras, _ = sensors(task)
    if (manifest.get("format") != "vista-batch-v1" or manifest["scene_version"] != SCENE_VERSION
            or task not in manifest["tasks"] or tuple(manifest["tasks"][task]["cameras"]) != cameras):
        raise ValueError(f"Incompatible batch or missing {task} at {batch}")
    records = []
    for seed, attempts in manifest["tasks"][task]["episodes"].items():
        exported = [a for a in attempts if a.get("hdf5")]
        if not exported:
            continue
        if len(exported) != 1:
            raise ValueError(f"Ambiguous demonstration for seed {seed}")
        a = exported[0]
        if a["seed"] != int(seed) or not a["success"] or a["status"] != "finished":
            raise ValueError(f"Invalid exported attempt at seed {seed}")
        records.append({k: a[k] for k in ("seed", "hdf5", "sha256", "ticks", "home_end_frame", "attempt")})
    if len(records) < 2:
        raise ValueError(f"{task} needs at least two exported episodes for a held-out split")
    records.sort(key=lambda r: r["seed"])
    # Cover earlier and later collection seeds even when the guide or effort evolves.
    rng = random.Random(42)
    count = max(1, round(.2 * len(records)))
    held_out = set()
    for i in range(count):
        group = records[i * len(records) // count : (i + 1) * len(records) // count]
        held_out.add(rng.choice(group)["seed"])
    return dict(batch=manifest["name"], task=task, scene_version=SCENE_VERSION,
                train=[r for r in records if r["seed"] not in held_out],
                validation=[r for r in records if r["seed"] in held_out])


class Demonstrations(Dataset):
    def __init__(self, batch, task, records, cfg, *, stats=None):
        self.cfg = cfg
        self.paths, self.states, self.actions = [], [], []
        cameras, joints = sensors(task)
        if tuple(cfg.cameras) != cameras or cfg.state_dim != len(joints):
            raise ValueError(f"Model sensors differ from task {task}")
        batch = Path(batch).resolve()
        for r in records:
            path = (batch / r["hdf5"]).resolve()
            ticks, seed, home = r["ticks"], r["seed"], r["home_end_frame"]
            if path.parent != batch / task or not path.is_file() or file_digest(path) != r["sha256"]:
                raise ValueError(f"Missing, moved or changed demonstration: {path}")
            with h5py.File(path, "r") as f:
                a = f.attrs
                if (a["task"] != task or a["scene_version"] != SCENE_VERSION or a["seed"] != seed
                        or a["fps"] != CONTROL_HZ or not 0 < home < ticks or a["home_end_frame"] != home
                        or tuple(json.loads(a["joint_names"])) != joints or tuple(json.loads(a["camera_names"])) != cameras
                        or json.loads(a["units"]) != (["rad"] * 5 + ["opening_0_1"]) * (len(joints) // 6)):
                    raise ValueError(f"Invalid demonstration metadata: {path}")
                state = f["observations/state"][:].astype(np.float32)
                action = f["action"][:].astype(np.float32)
                if (state.shape != (ticks, len(joints)) or action.shape != state.shape
                        or not np.isfinite(state).all() or not np.isfinite(action).all()):
                    raise ValueError(f"Invalid state/action sequence: {path}")
                for cam in cameras:
                    images = f[f"observations/images/{cam}"]
                    if images.shape != (ticks, 480, 640, 3) or images.dtype != np.uint8:
                        raise ValueError(f"Invalid {cam} images: {path}")
            self.paths.append(path)
            self.states.append(state)
            self.actions.append(action)
        if not self.paths:
            raise ValueError("Empty episode split")
        self.ends = np.cumsum([len(s) for s in self.states]).tolist()
        self.stats = {} if stats is None else stats
        for name, arrays in (("state", self.states), ("action", self.actions)):
            if stats is None:
                joined = np.concatenate(arrays, axis=0)
                self.stats[name + "_mean"] = joined.mean(axis=0).tolist()
                self.stats[name + "_std"] = joined.std(axis=0).clip(.01).tolist()
            for values in arrays:
                values -= np.asarray(self.stats[name + "_mean"], dtype=np.float32)
                values /= np.asarray(self.stats[name + "_std"], dtype=np.float32)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = bisect_right(self.ends, index)
        tick = index - (self.ends[ep - 1] if ep else 0)
        with h5py.File(self.paths[ep], "r") as f:
            rgb = np.stack([resize_rgb(f[f"observations/images/{name}"][tick], self.cfg.image_hw)
                            for name in self.cfg.cameras]).transpose(0, 3, 1, 2)
        chunk = np.zeros((self.cfg.chunk_size, self.cfg.state_dim), np.float32)
        labels = self.actions[ep][tick:tick + self.cfg.chunk_size]
        chunk[:len(labels)] = labels
        return rgb, self.states[ep][tick], chunk, np.arange(self.cfg.chunk_size) >= len(labels)
