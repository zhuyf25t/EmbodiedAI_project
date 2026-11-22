"""Evaluate a saved Lift checkpoint and replay actual actions for evidence videos."""
import argparse
from pathlib import Path
from tempfile import mkdtemp

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from act.eval import EXECUTION_HORIZON, SimPool, summary
from act.files import atomic_json, file_digest
from contracts import CAMERAS, CONTROL_HZ, JOINT_NAMES, JointPositions
from scene.env import TabletopEnv
from scene.lift import Lift


def record_video(env, episode, directory):
    seed = episode["seed"]
    path = directory / f"seed-{seed}-{'success' if episode['success'] else 'failure'}.mp4"
    obs, judge = env.reset(seed=seed)
    font = ImageFont.load_default(size=21)
    writer = imageio_ffmpeg.write_frames(str(path), (1280, 528), fps=CONTROL_HZ,
                                         codec="libx264", macro_block_size=1)
    writer.send(None)

    def draw():
        canvas = Image.new("RGB", (1280, 528), "#182028")
        for i, name in enumerate(CAMERAS):
            canvas.paste(Image.fromarray(obs.images[name]), (640 * i, 0))
        ink = ImageDraw.Draw(canvas)
        ink.text((12, 490), f"wrist   |   seed {seed}   |   tick {obs.frame_no}", font=font)
        ink.text((650, 490), f"front   |   bottom {1000*judge['cube_clearance_m']:.1f} mm   |   "
                 f"contacts {int(judge['fixed_contact'])}/{int(judge['moving_contact'])}", font=font)
        writer.send(np.asarray(canvas))

    try:
        draw()
        for row in episode["trace"]:
            obs, _, hit, timeout, judge = env.step(JointPositions(tuple(row["action"])))
            if (obs.frame_no != row["tick"] or judge["success"] != row["success"]
                    or judge["fixed_contact"] != row["fixed_contact"] or judge["moving_contact"] != row["moving_contact"]
                    or abs(judge["cube_clearance_m"] - row["cube_clearance_m"]) > 1e-7):
                raise RuntimeError(f"Replay diverged at seed {seed}, tick {obs.frame_no}")
            draw()
        if hit != episode["success"] or not (hit or timeout):
            raise RuntimeError("Video terminal outcome differs from the evaluated episode")
    finally:
        writer.close()
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("output/eval/lift"))
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.episodes) < 1 or args.seed < 0:
        parser.error("Expected a nonnegative seed and positive episode/worker counts")

    import torch
    from act.model import ACT, ModelConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**saved["config"])
    protocol = saved["protocol"]
    if (protocol["task"] != "lift" or tuple(protocol["cameras"]) != CAMERAS
            or tuple(protocol["joint_names"]) != JOINT_NAMES or protocol["fps"] != CONTROL_HZ
            or protocol["max_ticks"] != Lift.max_steps or protocol["execution_horizon"] != EXECUTION_HORIZON
            or tuple(cfg.cameras) != CAMERAS or cfg.chunk_size != protocol["chunk_size"]):
        raise ValueError("Checkpoint and environment protocols differ")
    seeds = list(range(args.seed, args.seed + args.episodes))
    excluded = {r["seed"] for r in saved["training_data"]} | set(protocol["validation_seeds"])
    if excluded.intersection(seeds):
        raise ValueError("Independent test seeds overlap training or validation")
    model = ACT(cfg, pretrained=False).to(device)
    model.load_state_dict(saved["model"])
    args.out.mkdir(parents=True, exist_ok=True)
    out = Path(mkdtemp(dir=args.out, prefix="run-"))
    with SimPool(min(args.workers, len(seeds)), cfg.image_hw) as pool:
        episodes = pool.evaluate(model, saved["stats"], seeds, device)
    report = summary(episodes)
    report.update(epoch=saved["epoch"], checkpoint_sha256=file_digest(args.checkpoint), protocol=protocol, videos=[])
    for e in episodes:
        atomic_json(out / f"seed-{e['seed']}.json", e)
    if args.video:
        with TabletopEnv(Lift) as env:
            for outcome in (True, False):
                examples = [e for e in episodes if e["success"] == outcome][:2]
                report["videos"].extend(record_video(env, e, out) for e in examples)
    atomic_json(out / "summary.json", report)
    print(f"{out}: {report['successes']}/{report['total']} = {report['success_rate']:.1%}")


if __name__ == "__main__":
    main()
