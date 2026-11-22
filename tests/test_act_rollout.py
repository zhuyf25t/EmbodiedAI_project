"""Exercise the public evaluator with real MuJoCo, ffmpeg, and untrained ACT weights."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import imageio_ffmpeg
import torch

from act.data import sensors
from act.model import ACT, ModelConfig
from act.train import inference_checkpoint
from contracts import CONTROL_HZ, SCENE_VERSION
from evaluation.rollout import parse_args, validate_checkpoint


def fixture(task):
    cameras, joints = sensors(task)
    cfg = ModelConfig(cameras=cameras, state_dim=len(joints), image_hw=(32, 32), chunk_size=8,
                      hidden_dim=32, dim_feedforward=64, nheads=4, encoder_layers=1,
                      decoder_layers=1, posterior_layers=1)
    stats = {key: [float(key.endswith("std"))] * len(joints)
             for key in ("state_mean", "state_std", "action_mean", "action_std")}
    protocol = dict(cameras=list(cameras), joint_names=list(joints), task=task, fps=CONTROL_HZ,
                    scene_version=SCENE_VERSION, chunk_size=8, execution_horizon=8, validation_seeds=[3])
    return dict(config=asdict(cfg), stats=stats, protocol=protocol, training_data=[dict(seed=0)],
                validation={}, epoch=0)


class RolloutTest(unittest.TestCase):
    def test_defaults_and_independent_test_seeds(self):
        args = parse_args(["--out", "run01", "--task", "sort"])
        self.assertEqual((args.seed, args.seed + args.episodes - 1, args.episodes, args.workers), (1000, 1099, 100, 10))
        saved = fixture("sort")
        self.assertEqual(validate_checkpoint(saved, "sort", range(1000, 1100)).state_dim, 12)
        for seed in (0, 3):
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_checkpoint(saved, "sort", [seed])
        with self.assertRaisesRegex(ValueError, "protocol"):
            validate_checkpoint(saved, "lift", [1000])
        saved["stats"]["action_std"][5] = 0
        with self.assertRaisesRegex(ValueError, "normalization"):
            validate_checkpoint(saved, "sort", [1000])

    def test_named_cli_records_every_seed_in_all_three_scenes(self):
        project = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": str(project), "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            for task, count, cap in (("lift", 3, 120), ("stack", 1, 300), ("sort", 1, 300)):
                with self.subTest(task=task):
                    saved = fixture(task)
                    model = ACT(ModelConfig(**saved["config"]), pretrained=False)
                    # Constant zero joint targets give physical timeouts at these seeds.
                    with torch.no_grad():
                        for parameter in model.action_head.parameters():
                            parameter.zero_()
                    saved["model"] = model.state_dict()
                    path = root / "output/act/smoke" / task / "best.pt"
                    path.parent.mkdir(parents=True)
                    torch.save(inference_checkpoint(saved), path)
                    cmd = [sys.executable, "-m", "evaluation.rollout", "--out", "smoke", "--task", task,
                           "--episodes", str(count), "--threads", "2"]
                    run = subprocess.run(cmd, cwd=root, env=env, capture_output=True, encoding="utf8", errors="replace", timeout=480)
                    self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
                    folder, = (root / "output/eval/smoke" / task).iterdir()
                    report = json.loads((folder / "summary.json").read_text())
                    self.assertEqual((report["total"], report["successes"], report["success_rate"]), (count, 0, 0))
                    self.assertEqual(report["workers"], min(count, 2))
                    with path.open("rb") as stream:
                        self.assertEqual(report["checkpoint_sha256"], hashlib.file_digest(stream, "sha256").hexdigest())
                    self.assertEqual([r["seed"] for r in report["episodes"]], list(range(1000, 1000 + count)))
                    self.assertEqual(sorted(report["videos"]), sorted(p.name for p in folder.glob("*.mp4")))
                    for row in report["episodes"]:
                        trace = json.loads((folder / f"seed-{row['seed']}.json").read_text())
                        self.assertFalse(row["final_judge"]["success"])
                        self.assertEqual((row["ticks"], row["reason"], row["video_frames"]), (cap, "timeout", cap + 1))
                        self.assertEqual(len(trace["trace"]), cap)
                        self.assertEqual(len(trace["trace"][0]["action"]), len(sensors(task)[1]))
                        self.assertEqual(trace["trace"][-1]["tick"], cap)
                        reader = imageio_ffmpeg.read_frames(str(folder / row["video"]))
                        try:
                            meta = next(reader)
                            self.assertEqual(meta["fps"], CONTROL_HZ)
                            self.assertEqual(meta["size"], (480 * len(sensors(task)[0]), 432))
                            self.assertEqual(sum(1 for frame in reader), cap + 1)
                        finally:
                            reader.close()
                    if task == "lift":
                        # Broken inference is an incomplete run, never an ordinary policy timeout.
                        with torch.no_grad():
                            model.action_head[-1].bias.fill_(float("nan"))
                        torch.save(inference_checkpoint(saved), path)
                        broken = subprocess.run(cmd[:cmd.index("--episodes")] + ["--episodes", "1"], cwd=root,
                                                env=env, capture_output=True, encoding="utf8", errors="replace", timeout=90)
                        self.assertNotEqual(broken.returncode, 0)
                        self.assertIn("Non-finite ACT joint target", broken.stderr)
                        other, = (d for d in folder.parent.iterdir() if d != folder)
                        self.assertFalse(list(other.glob("*.mp4")))
                        self.assertFalse((other / "summary.json").exists())
                        self.assertEqual(json.loads((folder / "summary.json").read_text()), report)
