"""Offline episode split, normalization, validation and portable checkpoints."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from act.data import sensors
from act.files import atomic_json, file_digest
from act.model import ACT, ModelConfig
from act.train import datasets, inference_checkpoint, parse_args, run_epoch, save_checkpoint
from contracts import SCENE_VERSION


def add_episode(batch, task, seed, manifest):
    cameras, names = sensors(task)
    ticks = 3 + seed % 3
    path = batch / task / f"ep_{seed:04d}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.full((ticks, len(names)), seed / 10., dtype="f4")
    with h5py.File(path, "w") as f:
        f.attrs.update(task=task, seed=seed, scene_version=SCENE_VERSION, fps=10,
                       home_end_frame=1, camera_names=json.dumps(cameras), joint_names=json.dumps(names),
                       units=json.dumps((["rad"]*5 + ["opening_0_1"]) * (len(names)//6)))
        f["observations/state"] = values
        f["action"] = values
        for cam in cameras:
            f.create_dataset("observations/images/"+cam, (ticks,480,640,3), chunks=(1,480,640,3), dtype="u1")
    record = dict(seed=seed, hdf5=path.relative_to(batch).as_posix(), sha256=file_digest(path),
                  status="finished", success=True, attempt=2, ticks=ticks, home_end_frame=1)
    spec = manifest["tasks"].setdefault(task, dict(cameras=list(cameras), episodes={}))
    spec["episodes"][str(seed)] = [dict(seed=seed, attempt=1, status="interrupted", success=False), record]
    atomic_json(batch/"manifest.json", manifest)


class ACTTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.batch = Path(tmp.name)/"run01"
        self.manifest = dict(format="vista-batch-v1", name="run01", scene_version=SCENE_VERSION, tasks={})
        for seed in range(5):
            add_episode(self.batch, "sort", seed, self.manifest)
        cameras, names = sensors("sort")
        self.cfg = ModelConfig(cameras=cameras, state_dim=len(names), image_hw=(32,32), chunk_size=4,
                               hidden_dim=32, nheads=4, dim_feedforward=64, encoder_layers=1,
                               decoder_layers=1, posterior_layers=1)

    def test_split_is_by_episode_and_frozen_across_additions(self):
        split, train, val = datasets(self.batch, "sort", self.cfg)
        train_ids = {r["seed"] for r in split["train"]}
        val_ids = {r["seed"] for r in split["validation"]}
        self.assertEqual((len(train_ids),len(val_ids)), (4,1))
        self.assertFalse(train_ids & val_ids)
        expected_mean = np.average([r["seed"]/10 for r in split["train"]], weights=[r["ticks"] for r in split["train"]])
        np.testing.assert_allclose(split["stats"]["action_mean"],expected_mean, atol=1e-7)
        self.assertEqual(train.stats,val.stats)
        rgb, state, chunk, pad = train[train.ends[0]-1]
        self.assertEqual((rgb.shape,state.shape), ((3,3,32,32),(12,)))
        self.assertEqual(pad.tolist(),[False,True,True,True])
        np.testing.assert_array_equal(chunk[1:],0)
        add_episode(self.batch, "sort", 7, self.manifest)
        same, reloaded, _ = datasets(self.batch, "sort", self.cfg, frozen=split)
        self.assertEqual(split,same)
        self.assertEqual(len(reloaded),len(train))
        path = self.batch/split["train"][0]["hdf5"]
        with path.open("ab") as f:
            f.write(b"unexpected data")
        with self.assertRaisesRegex(ValueError,"changed demonstration"):
            datasets(self.batch, "sort", self.cfg, frozen=split)

    def test_validation_has_repeatable_loss_and_checkpoint_can_infer(self):
        split, _, val = datasets(self.batch, "sort", self.cfg)
        loader = DataLoader(val, batch_size=2)
        model = ACT(self.cfg, pretrained=False)
        score = run_epoch(model, loader, torch.device("cpu"))
        self.assertEqual(score, run_epoch(model, loader, torch.device("cpu")))
        self.assertAlmostEqual(score["loss"], score["l1"] + 10*score["kl"])
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        from dataclasses import asdict
        saved = dict(epoch=1, config=asdict(self.cfg), protocol=dict(task="sort"),
                     model=model.state_dict(), stats=val.stats, training_data=split["train"], validation=score)
        path = self.batch/"best.pt"
        save_checkpoint(path, inference_checkpoint(saved))
        restored = torch.load(path, weights_only=True, map_location="cpu")
        copy = ACT(ModelConfig(**restored["config"]), pretrained=False).eval()
        copy.load_state_dict(restored["model"])
        images, states, _, _ = next(iter(loader))
        with torch.inference_mode():
            torch.testing.assert_close(model(images,states)[0], copy(images,states)[0])

    def test_two_required_cli_arguments(self):
        args = parse_args(["--out", "run02", "--task", "stack"])
        self.assertEqual((args.out,args.task,args.data),("run02","stack",Path("data/run02")))
        second = parse_args(["--out", "experiment2", "--task", "lift", "--data", "data/run01"])
        self.assertEqual((second.out, second.data), ("experiment2", Path("data/run01")))


if __name__ == "__main__":
    unittest.main()
