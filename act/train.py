"""Train Lift ACT and select weights solely by closed-loop validation success."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path

from .eval import EXECUTION_HORIZON, VALIDATION_SEEDS, SimPool, summary
from .files import atomic_json


def save_checkpoint(path, value):
    import torch
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def inference_checkpoint(saved):
    return {key: saved[key] for key in ("epoch", "config", "protocol", "model", "stats", "training_data", "validation")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=10, help="Concurrent validation environments")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.workers, args.batch_size) < 1:
        parser.error("epochs, workers and batch-size must be positive")

    import torch
    from torch.utils.data import DataLoader
    from contracts import CONTROL_HZ, JOINT_NAMES
    from scene.lift import Lift
    from .data import Demonstrations
    from .model import ACT, ModelConfig, objective

    if not torch.cuda.is_available():
        parser.error("GPU training requires a CUDA-enabled PyTorch installation and an available GPU")
    torch.manual_seed(42)
    device = torch.device("cuda")
    cfg = ModelConfig()
    dataset = Demonstrations(args.data, cfg, VALIDATION_SEEDS)
    protocol = dict(task="lift", fps=CONTROL_HZ, joint_names=list(JOINT_NAMES), cameras=list(cfg.cameras),
                    validation_seeds=list(VALIDATION_SEEDS), max_ticks=Lift.max_steps,
                    chunk_size=cfg.chunk_size, execution_horizon=EXECUTION_HORIZON,
                    success="both jaws in force contact and cube bottom >= 50 mm above tabletop",
                    best_by="successful episodes, earliest epoch on tie")
    cfg_dict = asdict(cfg)
    if args.resume:
        saved = torch.load(args.out / "last.pt", map_location="cpu", weights_only=True)
        if (saved["protocol"] != protocol or saved["training_data"] != dataset.records
                or saved["config"] != cfg_dict or saved["batch_size"] != args.batch_size):
            raise ValueError("Resume requires the original data, model, batch size and validation protocol")
    else:
        saved = None
        if args.out.exists():
            raise FileExistsError(f"Choose a fresh experiment directory or use --resume: {args.out}")
    model = ACT(cfg, pretrained=not args.resume).to(device)
    backbone = list(model.backbone.parameters())
    trunk = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": 1e-5}, {"params": trunk, "lr": 1e-4}],
                                  weight_decay=1e-4)
    first, best, best_epoch = 1, -1, 0
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], device)
        first = saved["epoch"] + 1
        best = saved["best_successes"]
        best_epoch = saved["best_epoch"]
        if best_epoch == saved["epoch"]:
            # last.pt is committed first; repair an interrupted best.pt publication.
            save_checkpoint(args.out / "best.pt", inference_checkpoint(saved))
        elif not (args.out / "best.pt").is_file():
            raise FileNotFoundError("The earlier winning checkpoint is missing from the experiment")
        del saved
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        atomic_json(args.out / "training_data.json", dict(episodes=dataset.records, stats=dataset.stats))
        atomic_json(args.out / "protocol.json", protocol)
    if first > args.epochs:
        return
    with SimPool(min(args.workers, len(VALIDATION_SEEDS)), cfg.image_hw) as pool:
        for epoch in range(first, args.epochs + 1):
            model.train()
            totals = torch.zeros(3, device=device)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                generator=torch.Generator().manual_seed(42 + epoch))
            for batch in loader:
                rgb, state, action, pad = [v.to(device, non_blocking=True) for v in batch]
                pred, mean, logvar = model(rgb, state, action, pad)
                loss, l1, kl = objective(pred, action, pad, mean, logvar)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                totals += torch.stack((loss.detach(), l1.detach(), kl.detach())) * len(state)
            validation = summary(pool.evaluate(model, dataset.stats, VALIDATION_SEEDS, device))
            metrics = dict(zip(("loss", "l1", "kl"), (totals / len(dataset)).tolist()))
            atomic_json(args.out / f"epoch_{epoch:03d}.json", dict(epoch=epoch, training=metrics, validation=validation))
            weights = dict(epoch=epoch, config=cfg_dict, protocol=protocol, model=model.state_dict(),
                           stats=dataset.stats, training_data=dataset.records, validation=validation)
            score = validation["successes"]
            if score > best:
                best, best_epoch = score, epoch
            save_checkpoint(args.out / "last.pt", dict(**weights, optimizer=optimizer.state_dict(), best_successes=best,
                            best_epoch=best_epoch, batch_size=args.batch_size,
                            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device)))
            if epoch == best_epoch:
                save_checkpoint(args.out / "best.pt", weights)
            print(f"epoch {epoch}/{args.epochs}  loss {metrics['loss']:.4f}  validation {score}/20  best {best}/20", flush=True)


if __name__ == "__main__":
    main()
