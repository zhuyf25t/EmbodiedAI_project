"""Evaluate a trained ACT policy in closed loop and record every test episode."""
import argparse
import hashlib
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

from act.eval import SimPool, summary
from act.files import atomic_json
from collection.batch import batch_name
from contracts import CONTROL_HZ, SCENE_VERSION


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Training experiment name, e.g. run01")
    parser.add_argument("--task", choices=("lift", "stack", "sort"))
    parser.add_argument("--seed", type=int, default=1000, help="First test seed (default 1000)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of test seeds (default 100)")
    parser.add_argument("--threads", "--workers", dest="workers", type=int, default=10,
                        help="Parallel simulation processes (default 10)")
    # Compatibility with the original Lift-only command that accepted a file and output path.
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--video", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.workers, args.episodes) < 1 or args.seed < 0:
        parser.error("Expected a nonnegative start seed and positive episode and worker counts")
    if args.checkpoint is None:
        if not args.out or not args.task:
            parser.error("--out and --task are required")
        try:
            batch_name(args.out)
        except ValueError as error:
            parser.error(str(error))
    return args


def validate_checkpoint(saved, task, seeds):
    from act.data import sensors
    from act.model import ModelConfig
    from scene.tasks import TASKS

    cfg = ModelConfig(**saved["config"])
    protocol = saved["protocol"]
    cameras, joints = sensors(task)
    horizon = protocol["execution_horizon"]
    if (protocol["scene_version"] != SCENE_VERSION or protocol["task"] != task
            or tuple(protocol["cameras"]) != cameras or tuple(cfg.cameras) != cameras
            or tuple(protocol["joint_names"]) != joints or cfg.state_dim != len(joints)
            or protocol["fps"] != CONTROL_HZ or cfg.chunk_size != protocol["chunk_size"]
            or type(horizon) is not int or not 1 <= horizon <= cfg.chunk_size
            or protocol.get("max_ticks", TASKS[task].max_steps) != TASKS[task].max_steps):
        raise ValueError("Checkpoint and environment protocols differ")
    for key in ("state_mean", "state_std", "action_mean", "action_std"):
        value = np.asarray(saved["stats"][key])
        if value.shape != (len(joints),) or not np.isfinite(value).all() or (key.endswith("_std") and np.any(value <= 0)):
            raise ValueError(f"Invalid checkpoint normalization: {key}")
    excluded = {row["seed"] for row in saved["training_data"]} | set(protocol["validation_seeds"])
    overlap = sorted(excluded.intersection(seeds))
    if overlap:
        raise ValueError(f"Test seeds overlap training or validation: {overlap}")
    return cfg


def evaluate(args):
    if args.checkpoint is None:
        checkpoint = Path("output/act") / args.out / args.task / "best.pt"
        base = Path("output/eval") / args.out / args.task
    else:
        checkpoint = args.checkpoint
        base = Path(args.out) if args.out else Path("output/eval") / (args.task or "lift")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing ACT checkpoint: {checkpoint}")

    import torch
    from act.model import ACT
    from scene.tasks import TASKS

    # Read the hash and the parameters from the same open file if training publishes a new best.
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
    task = args.task or saved["protocol"]["task"]
    seeds = range(args.seed, args.seed + args.episodes)
    cfg = validate_checkpoint(saved, task, seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ACT(cfg, pretrained=False)
    model.load_state_dict(saved.pop("model"))
    model.to(device)
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(mkdtemp(dir=base, prefix=f"seeds-{seeds[0]}-{seeds[-1]}-"))
    workers = min(args.workers, len(seeds))
    horizon = saved["protocol"]["execution_horizon"]
    print(f"{task} | {checkpoint} | {device} | {workers} workers | {directory}", flush=True)
    with SimPool(workers, cfg.image_hw, task, directory) as pool:
        episodes = pool.evaluate(model, saved["stats"], seeds, device, horizon)
    report = summary(episodes)
    report.update(task=task, seed_start=seeds[0], seed_end=seeds[-1], workers=workers,
                  epoch=saved["epoch"], checkpoint_sha256=digest, protocol=saved["protocol"],
                  max_ticks=TASKS[task].max_steps, videos=[e["video"] for e in episodes])
    atomic_json(directory / "summary.json", report)
    print(f"{directory}: {report['successes']}/{report['total']} = {report['success_rate']:.1%}", flush=True)
    return directory


def main():
    args = parse_args()
    try:
        evaluate(args)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
