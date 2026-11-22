"""Exercise the collector entry point without a remote vision model."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from langchain_core.messages import AIMessage

from brains.vista.file_lock import FileLock
from collection.batch import attempt_path, file_hash, logs_for, read_manifest, save_attempt
from run_collect import parse_args, run_batch, same_provider


class OfflineModel:
    model = "offline-test"

    def close(self):
        pass

    def invoke(self, messages, *, tools, system):
        frame = next(m.additional_kwargs["vista_images"][0]["frame_id"] for m in reversed(messages)
                     if m.additional_kwargs.get("vista_images"))
        if any(t["function"]["name"] == "finish" for t in tools):
            name, args = "finish", dict(success=False, reason="offline check")
        else:
            name, args = "play", dict(basis_frame_id=frame, expectation="hold",
                                      actions=[dict(action="observe", args={})])
        return AIMessage(content="", tool_calls=[dict(name=name, args=args, id=f"{name}_{frame}")])


class OfflineInterruption(OfflineModel):
    def invoke(self, messages, **options):
        raise RuntimeError("test rate limit")


class CollectionBatchTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.data = self.root / "data/run01"

    def collect(self, arguments, model=OfflineModel):
        return run_batch(parse_args(arguments.split()), root=self.root, model_factory=model)

    def test_resume_can_change_reasoning_effort_and_keeps_other_provider_options(self):
        before = dict(model="m1", api="responses", request_options={"reasoning_effort": "xhigh", "temperature": 0.1})
        after = {**before, "request_options": {"reasoning_effort": "high", "temperature": 0.1}}
        self.assertTrue(same_provider(before, after))
        self.assertFalse(same_provider(before, {**after, "model": "m2"}))
        self.assertFalse(same_provider(before, {**after, "api": "chat_completions"}))
        self.assertFalse(same_provider(before, {**after, "request_options": {"temperature": 0.2}}))
        self.assertEqual(before["request_options"]["reasoning_effort"], "xhigh")

    def test_resume_add_fix_and_parallel_task_isolation(self):
        with self.assertRaises(RuntimeError):
            self.collect("--out run01 --task lift --seed 3 --max-play-rounds 1", OfflineInterruption)
        first = attempt_path(self.data, "lift", 3, 1)
        original = {p: file_hash(p) for p in first.rglob("*") if p.is_file()}
        self.collect("--resume run01")
        self.assertFalse(attempt_path(self.data, "lift", 3, 2).exists())
        with self.assertRaises(FileExistsError):
            self.collect("--out run01 --task sort")
        with FileLock(logs_for(self.data).parent/".locks/run01.lock", timeout=0):
            with self.assertRaises(BlockingIOError):
                self.collect("--resume run01 --add 1")
        self.collect("--resume run01 --fix --add 1 --threads 2")
        self.collect("--resume run01 --fix")
        manifest = read_manifest(self.data)
        rows = manifest["tasks"]["lift"]["episodes"]
        self.assertEqual(manifest["seeds"], [3, 4])
        self.assertEqual([a["status"] for a in rows["3"]], ["interrupted", "finished"])
        self.assertEqual(len(rows["4"]), 1)
        self.assertEqual(rows["3"][1]["reason"], "play_round_limit")
        self.assertEqual(rows["4"][0]["reason"], "play_round_limit")
        self.assertTrue(all(file_hash(p) == digest for p,digest in original.items()))
        self.collect("--out multi --task lift sort --max-play-rounds 1 --threads 2")
        self.collect("--resume multi --add 1 --threads 2")
        tasks = read_manifest(self.root/"data/multi")["tasks"]
        self.assertEqual(list(tasks), ["lift", "sort"])
        self.assertTrue(all(list(t["episodes"]) == ["0", "1"] and t["summary"]["attempts"] == 2
                            for t in tasks.values()))

    def test_bare_resume_only_starts_pending_seeds_and_fix_recovers_dead_worker(self):
        with self.assertRaises(RuntimeError):
            self.collect("--out run01 --task lift --episodes 2 --max-play-rounds 1", OfflineInterruption)
        self.collect("--resume run01")
        rows = read_manifest(self.data)["tasks"]["lift"]["episodes"]
        self.assertEqual([row["status"] for row in rows["0"]], ["interrupted"])
        self.assertEqual([row["status"] for row in rows["1"]], ["finished"])
        # Reproduce an abandoned manifest row and a worker which still owns its lock.
        previous = rows["0"][0]
        previous["status"] = "running"
        save_attempt(self.data, "lift", previous)
        with FileLock(logs_for(self.data) / "lift/.ep_0000.lock", timeout=0):
            self.collect("--resume run01 --fix")
        self.assertEqual(len(read_manifest(self.data)["tasks"]["lift"]["episodes"]["0"]), 1)
        self.collect("--resume run01 --fix")
        attempts = read_manifest(self.data)["tasks"]["lift"]["episodes"]["0"]
        self.assertEqual([a["status"] for a in attempts], ["interrupted", "finished"])
        self.assertEqual(attempts[0]["reason"], "worker stopped before finalization")

    def test_export_repair_and_existing_file_are_write_once(self):
        self.collect("--out run01 --task lift --max-play-rounds 1")
        entry = read_manifest(self.data)["tasks"]["lift"]["episodes"]["0"][0]
        raw = attempt_path(self.data, "lift", 0, 1)/"raw"
        # Construct an export-only fixture using a completed tick and both cameras.
        ticks = (raw/"ticks.jsonl").read_text("utf8").splitlines()
        last = json.loads(ticks[-1])
        last["judge_after"]["success"] = True
        (raw/"ticks.jsonl").write_text("\n".join(ticks[:-1]+[json.dumps(last)])+"\n", "utf8")
        entry.update(reason="success", success=True, export_error="disk full")
        save_attempt(self.data, "lift", entry)
        self.collect("--resume run01 --fix")
        success = read_manifest(self.data)["tasks"]["lift"]["episodes"]["0"][0]
        h5 = self.data/success["hdf5"]
        digest = file_hash(h5)
        self.assertEqual(digest, success["sha256"])
        mtime = h5.stat().st_mtime_ns
        success.pop("hdf5")  # Crash between atomic publication and manifest update.
        save_attempt(self.data, "lift", success)
        self.collect("--resume run01 --fix")
        self.assertEqual((file_hash(h5),h5.stat().st_mtime_ns), (digest,mtime))
        h5.write_bytes(b"unrelated fixture")
        success.pop("hdf5", None)
        save_attempt(self.data, "lift", success)
        with self.assertRaises(RuntimeError):
            self.collect("--resume run01 --fix")
        self.assertEqual(h5.read_bytes(), b"unrelated fixture")
        self.assertEqual(len(read_manifest(self.data)["tasks"]["lift"]["episodes"]["0"]), 1)

    def test_invalid_cli(self):
        for args in ("--out ../x --task lift", "--out a --task lift --fix", "--resume a --task lift",
                     "--resume a --episodes 2", "--resume a --seed 0", "--resume a --add 0",
                     "--out a --resume b --task lift", "--out a --task lift lift"):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args.split())
