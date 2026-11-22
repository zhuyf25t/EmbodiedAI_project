"""Train a task-specific ACT checkpoint; select best by held-out L1 + 10 * KL."""
import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path

from brains.vista.file_lock import FileLock
from collection.batch import batch_name
from contracts import CONTROL_HZ, SCENE_VERSION
from .eval import EXECUTION_HORIZON
from .files import atomic_json, read_json


def save_checkpoint(path, value):
    import torch
    temp = path.with_suffix(".tmp")
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def inference_checkpoint(saved):
    return {key: saved[key] for key in (
        "epoch", "config", "protocol", "model", "stats", "training_data", "validation")}


def run_epoch(model, loader, device, optimizer=None):
    """L1 is averaged over real action coordinates, KL over posterior examples."""
    import torch
    from .model import KL_WEIGHT, objective

    training = optimizer is not None
    model.train(training)
    fixed_noise = None if training else torch.Generator(device=device).manual_seed(42)
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    with torch.set_grad_enabled(training):
        for batch in loader:
            rgb, state, action, pad = (v.to(device, non_blocking=True) for v in batch)
            pred, mean, logvar = model(rgb, state, action, pad, generator=fixed_noise)
            loss, l1, kl = objective(pred, action, pad, mean, logvar)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            coordinates = (~pad).sum() * action.shape[-1]
            totals[0] += l1.detach().double() * coordinates
            totals[1] += coordinates
            totals[2] += kl.detach().double() * len(state)
            totals[3] += len(state)
    l1_sum, coords, kl_sum, samples = totals.tolist()
    l1, kl = l1_sum / coords, kl_sum / samples
    result = dict(loss=l1 + KL_WEIGHT * kl, l1=l1, kl=kl)
    if not all(math.isfinite(x) for x in result.values()):
        raise FloatingPointError(f"Non-finite epoch metric: {result}")
    return result


def datasets(batch, task, cfg, *, frozen=None):
    from .data import Demonstrations, snapshot
    split = frozen if frozen is not None else snapshot(batch, task)
    if split["task"] != task or split["scene_version"] != SCENE_VERSION:
        raise ValueError("Frozen snapshot has incompatible task or scene version")
    seeds = [r["seed"] for r in split["train"] + split["validation"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seed in frozen episode split")
    train = Demonstrations(batch, task, split["train"], cfg, stats=split.get("stats"))
    validation = Demonstrations(batch, task, split["validation"], cfg, stats=train.stats)
    return {**split, "stats": train.stats}, train, validation


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=batch_name, help="Experiment name under output/act/")
    parser.add_argument("--task", required=True, choices=("lift", "stack", "sort"))
    parser.add_argument("--data", type=Path, help="Override the input batch (default: data/<out>)")
    parser.add_argument("--epochs", type=int, default=100, help="Total epoch target, including on resume")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2, help="HDF5/image loader processes per split")
    parser.add_argument("--resume", action="store_true", help="Continue the saved split and weights")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size) < 1 or args.workers < 0:
        parser.error("epochs and batch size must be positive; workers must be nonnegative")
    if args.data is None:
        args.data = Path("data") / args.out
    return args


def train(args, device):
    import torch
    from torch.utils.data import DataLoader, RandomSampler
    from .data import sensors
    from .model import ACT, ModelConfig

    if device.type != "cuda":
        raise ValueError("Training requires a CUDA device")
    batch = args.data
    if batch.parent == Path(".") and not (batch / "manifest.json").is_file():
        batch = Path("data") / batch
    if batch.name == args.task and not (batch / "manifest.json").is_file():
        batch = batch.parent
    manifest = read_json(batch / "manifest.json")
    if manifest.get("format") != "vista-batch-v1" or manifest.get("scene_version") != SCENE_VERSION:
        raise ValueError("Expected a matching VISTA batch manifest")
    out = Path("output/act") / args.out / args.task
    if out.exists() and not args.resume:
        raise FileExistsError(f"Experiment already exists at {out}; use --resume or a new name")
    cameras, joints = sensors(args.task)
    cfg = ModelConfig(state_dim=len(joints), cameras=cameras)
    config = asdict(cfg)
    frozen = read_json(out / "training_data.json") if args.resume else None
    if frozen is not None and frozen["batch"] != manifest["name"]:
        raise ValueError("Resume requires the original batch")
    split, train_set, val_set = datasets(batch, args.task, cfg, frozen=frozen)
    protocol = dict(scene_version=SCENE_VERSION, task=args.task, cameras=list(cameras), joint_names=list(joints),
                    fps=CONTROL_HZ, chunk_size=cfg.chunk_size, execution_horizon=EXECUTION_HORIZON,
                    validation_seeds=[r["seed"] for r in split["validation"]],
                    best_by="lowest held-out L1 + 10 * KL, earliest epoch on tie")
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    with FileLock(out / ".training.lock", timeout=0):
        torch.manual_seed(42)
        if not args.resume:
            atomic_json(out / "training_data.json", {**split, "batch_size": args.batch_size})
        elif split["batch_size"] != args.batch_size:
            raise ValueError("Resume requires the original batch size")
        last = out / "last.pt"
        saved = torch.load(last, map_location="cpu", weights_only=True) if args.resume and last.is_file() else None
        if saved is not None:
            if (saved["protocol"] != protocol or saved["config"] != config or saved["stats"] != train_set.stats
                    or saved["training_data"] != split["train"] or saved["validation_data"] != split["validation"]):
                raise ValueError("Checkpoint differs from the frozen training snapshot")
        elif args.resume and (out / "best.pt").exists():
            raise FileNotFoundError("last.pt is missing from this experiment")

        model = ACT(cfg, pretrained=saved is None).to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": [p for n,p in model.named_parameters() if not n.startswith("backbone.")], "lr": 1e-4},
        ], weight_decay=1e-4)
        first, best, best_epoch = 1, math.inf, 0
        if saved is not None:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            torch.set_rng_state(saved["rng"])
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
            first = saved["epoch"] + 1
            best, best_epoch = saved["best_loss"], saved["best_epoch"]
            if best_epoch == saved["epoch"]:
                save_checkpoint(out / "best.pt", inference_checkpoint(saved))
            elif not (out / "best.pt").is_file():
                raise FileNotFoundError("Missing earlier best.pt")
            del saved

        order = torch.Generator()
        options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        if args.workers:
            options.update(persistent_workers=True, prefetch_factor=1)
        train_loader = DataLoader(train_set, sampler=RandomSampler(train_set, generator=order),
                                  generator=torch.Generator().manual_seed(42), **options)
        val_loader = DataLoader(val_set, shuffle=False, generator=torch.Generator().manual_seed(42), **options)
        print(f"{out}: {len(split['train'])} train / {len(split['validation'])} held-out episodes", flush=True)
        for epoch in range(first, args.epochs + 1):
            order.manual_seed(42 + epoch)
            training = run_epoch(model, train_loader, device, optimizer)
            validation = run_epoch(model, val_loader, device)
            if validation["loss"] < best:
                best, best_epoch = validation["loss"], epoch
            weights = dict(epoch=epoch, config=config, protocol=protocol, model=model.state_dict(),
                           stats=train_set.stats, training_data=split["train"], validation=validation)
            atomic_json(out / f"epoch_{epoch:03d}.json", dict(epoch=epoch, training=training, validation=validation))
            save_checkpoint(last, dict(**weights, validation_data=split["validation"], optimizer=optimizer.state_dict(),
                                       best_loss=best, best_epoch=best_epoch, rng=torch.get_rng_state(),
                                       cuda_rng=torch.cuda.get_rng_state(device)))
            if epoch == best_epoch:
                save_checkpoint(out / "best.pt", weights)
            print(f"epoch {epoch}/{args.epochs}  train {training['loss']:.5f}  "
                  f"val {validation['loss']:.5f}  best {best:.5f} (epoch {best_epoch})", flush=True)


def main():
    args = parse_args()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("ACT training requires a CUDA GPU and CUDA-enabled PyTorch.")
    train(args, torch.device("cuda"))


if __name__ == "__main__":
    main()
