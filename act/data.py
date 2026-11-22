"""Frame-aligned supervision. No chunk or normalization statistic crosses a split."""
from bisect import bisect_right
import json
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset

from contracts import CAMERAS, CONTROL_HZ, JOINT_NAMES
from .files import file_digest, read_json
from .images import resize_rgb

class Demonstrations(Dataset):
    def __init__(self, roots, cfg, validation_seeds):
        self.cfg = cfg
        self.records, self.paths, self.states, self.actions = [], [], [], []
        seeds = set()
        for root in map(Path, roots):
            manifest = read_json(root / "manifest.json")
            if manifest["task"] != "lift" or tuple(manifest["cameras"]) != CAMERAS:
                raise ValueError(f"Incompatible task or cameras in {root}")
            for entry in manifest["episodes"]:
                if not entry.get("success"):
                    continue
                seed = entry["seed"]
                if seed in seeds or seed in validation_seeds or entry["status"] != "finished":
                    raise ValueError(f"Duplicate, held-out or incomplete demonstration: seed {seed}")
                path = (root / entry["hdf5"]).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError(f"Episode outside dataset: {path}")
                sha = file_digest(path)
                if entry.get("sha256", sha) != sha:
                    raise ValueError(f"File identity changed: {path}")
                with h5py.File(path, "r") as h:
                    a = h.attrs
                    if (a["task"] != "lift" or a["seed"] != seed or a["fps"] != CONTROL_HZ
                            or tuple(json.loads(a["joint_names"])) != JOINT_NAMES
                            or tuple(json.loads(a["camera_names"])) != CAMERAS
                            or json.loads(a["units"]) != ["rad"] * 5 + ["opening_0_1"]):
                        raise ValueError(f"Invalid sensor/action contract: {path}")
                    s = h["observations/state"][:].astype(np.float32)
                    u = h["action"][:].astype(np.float32)
                    if (u.shape != (entry["ticks"], cfg.state_dim) or not len(u) or s.shape != u.shape
                            or not np.isfinite(u).all() or not np.isfinite(s).all()):
                        raise ValueError(f"Invalid state/action frames: {path}")
                    for name in CAMERAS:
                        stream = h[f"observations/images/{name}"]
                        if stream.shape != (len(s), 480, 640, 3) or stream.dtype != np.uint8:
                            raise ValueError(f"Invalid camera {name}: {path}")
                self.records.append(dict(seed=seed, hdf5=entry["hdf5"], dataset=str(root),
                                         ticks=len(s), sha256=sha))
                seeds.add(seed)
                self.states.append(s)
                self.actions.append(u)
                self.paths.append(path)
        if not self.paths:
            raise ValueError("No successful demonstrations")
        self.ends = np.cumsum([len(s) for s in self.states]).tolist()
        self.stats = {}
        for key, arrays in (("state", self.states), ("action", self.actions)):
            joined = np.concatenate(arrays)
            mean, std = joined.mean(axis=0), joined.std(axis=0).clip(.01)
            self.stats[key + "_mean"] = mean.tolist()
            self.stats[key + "_std"] = std.tolist()
            for array in arrays:
                array -= mean
                array /= std

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = bisect_right(self.ends, index)
        t = index - (self.ends[ep - 1] if ep else 0)
        with h5py.File(self.paths[ep], "r") as h:
            rgb = np.stack([resize_rgb(h[f"observations/images/{name}"][t], self.cfg.image_hw)
                            for name in self.cfg.cameras]).transpose(0, 3, 1, 2)
        chunk = np.zeros((self.cfg.chunk_size, self.cfg.state_dim), np.float32)
        labels = self.actions[ep][t:t + len(chunk)]
        chunk[:len(labels)] = labels
        return rgb, self.states[ep][t], chunk, np.arange(len(chunk)) >= len(labels)
