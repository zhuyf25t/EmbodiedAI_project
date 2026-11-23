"""Named collection batches. All paths in the manifest are relative to the batch."""
import hashlib
import json
import re

from brains.vista.file_lock import FileLock
from collection.recorder import save_json
from contracts import CAMERAS, DUAL_CAMERAS, SCENE_VERSION


def batch_name(value):
    if (not re.fullmatch(r"[\w-][\w.-]*", value) or value.endswith(".")
            or value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL",
                *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}):
        raise ValueError("Use one directory name for the batch, e.g. run01")
    return value


def logs_for(data):
    return data.parent.parent / "output" / "vista" / data.name


def attempt_path(data, task, seed, number):
    return logs_for(data) / task / f"ep_{seed:04d}" / f"attempt_{number:02d}"


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_manifest(data):
    manifest = json.loads((data / "manifest.json").read_text("utf8"))
    if manifest.get("format") != "vista-batch-v1" or manifest["name"] != data.name:
        raise ValueError(f"Expected a named VISTA batch at {data}")
    return manifest


def write_manifest(data, manifest):
    for task in manifest["tasks"].values():
        rows = task["episodes"].values()
        attempts = [a for row in rows for a in row]
        task["summary"] = dict(planned=len(rows), started=sum(bool(r) for r in rows),
            attempts=len(attempts), first_success=sum(bool(r and r[0].get("success")) for r in rows),
            interrupted=sum(a["status"] == "interrupted" for a in attempts),
            exported=sum(any(a.get("hdf5") for a in r) for r in rows))
    save_json(data / "manifest.json", manifest)


def new_manifest(name, tasks, seeds, max_play, provider):
    return dict(format="vista-batch-v1", name=name, scene_version=SCENE_VERSION,
                source="vista", max_play_rounds=max_play, provider=provider, seeds=list(seeds),
                tasks={task: dict(cameras=list(DUAL_CAMERAS if task == "sort" else CAMERAS),
                                  episodes={str(n): [] for n in seeds}) for task in tasks})


def provider_for_seed(manifest, seed):
    for change in reversed(manifest.get("provider_changes", [])):
        if seed >= change["first_seed"]:
            return change["provider"]
    return manifest["provider"]


def extend(manifest, count, provider=None):
    first = manifest["seeds"][-1] + 1
    if provider is not None and provider != provider_for_seed(manifest, first - 1):
        manifest.setdefault("provider_changes", []).append(dict(first_seed=first, provider=provider))
    new = range(first, first + count)
    manifest["seeds"].extend(new)
    for task in manifest["tasks"].values():
        task["episodes"].update({str(n): [] for n in new})


def jobs(data, manifest, fix):
    for seed in manifest["seeds"]:
        for task, spec in manifest["tasks"].items():
            attempts = spec["episodes"][str(seed)]
            if not attempts:
                yield task, seed, "record"
                continue
            last = attempts[-1]
            if last.get("success"):
                if fix:
                    if not last.get("hdf5") or not (data / last["hdf5"]).is_file():
                        yield task, seed, "export"
                    elif last.get("sha256") and file_hash(data / last["hdf5"]) != last["sha256"]:
                        raise ValueError(f"Existing HDF5 checksum differs: {last['hdf5']}")
            elif fix and last["status"] in ("interrupted", "running"):
                yield task, seed, "record"


def save_attempt(data, task, entry):
    with FileLock(logs_for(data) / ".manifest.lock"):
        manifest = read_manifest(data)
        attempts = manifest["tasks"][task]["episodes"][str(entry["seed"])]
        if len(attempts) + 1 == entry["attempt"]:
            attempts.append(entry)
        elif len(attempts) == entry["attempt"]:
            attempts[-1] = entry
        else:
            raise RuntimeError("Attempt number changed while recording")
        write_manifest(data, manifest)
