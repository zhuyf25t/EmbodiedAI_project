"""Collect a seeded batch of tabletop demonstrations with VISTA."""
import argparse
import sys
import time
from tempfile import TemporaryDirectory
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

from langchain_core.messages import HumanMessage

from brains.vista.brain import VistaBrain
from brains.vista.file_lock import FileLock
from brains.vista.model import ChatModel
from brains.vista.settings import VistaSettings
from brains.vista.state import action_result, batch_status
from collection.batch import (attempt_path, batch_name, extend, file_hash, jobs, logs_for,
                              new_manifest, read_manifest, save_attempt, write_manifest)
from collection.recorder import EpisodeRecorder, export_demo, save_json
from contracts import DUAL_JOINT_NAMES, JOINT_NAMES, ObservationWindow, SCENE_VERSION
from control.vista import action_schemas, ArmExecutor
from scene.env import TabletopEnv
from scene.lift import Lift
from scene.front_ruler import front_ruler, sample_marks
from scene.stack import Stack
from scene.sort import Sort
from scene.live_view import launch_viewer

TASKS = {"lift": Lift, "stack": Stack, "sort": Sort}
MAX_PLAY_ROUNDS = 20


def execute_batch(brain, executor, actions, call):
    frames = [executor.obs]
    saved_tick = executor.tick

    def tick(before, target, after, info):
        frames.append(after)
        saved_tick(before, target, after, info)

    results = []
    stopped = False
    end_reason = None
    executor.tick = tick
    try:
        for i, action in enumerate(actions, 1):
            result = action_result(i, action, "not_executed")
            results.append(result)
            if stopped or executor.env.done:
                continue
            start = executor.obs.frame_no
            code = executor.execute(action["action"], action["args"])
            end = executor.obs.frame_no
            result.update(frame_range=[start, end], status="failed" if code else "completed",
                          action_applied=end > start, error={"code": code} if code else None)
            settled = code in (None, "GRASP_NOT_CONFIRMED")
            brain.storage.event("control_boundary", action=action["action"], start=start, end=end,
                                stable_frames=[end - 1, end] if settled else [], error=code)
            if code is None and executor.info["success"]:
                end_reason = "success"
            elif executor.env.done:
                end_reason = "timeout"
            if end_reason:
                executor.env.done = True
            stopped = bool(code)
        receipt = dict(batch_id=brain.active_batch_id, tool_call_id=call["id"], action_results=results,
                       execution_status=batch_status(results, cancelled=executor.env.done))
        brain.accept_batch_result(receipt, ObservationWindow(tuple(frames)))
        return end_reason
    finally:
        executor.tick = saved_tick


def collect_episode(env, brain, data, entry, persist, display, max_play_rounds):
    seed = entry["seed"]
    ep = f"ep_{seed:04d}"
    task = type(env.task).__name__.lower()
    raw = attempt_path(data, task, seed, entry["attempt"]) / "raw"
    obs, initial = env.reset(seed=seed, finish_on_success=False, limit_ticks=False)
    entry.update(status="running", started=True, success=False, ticks=0, play_rounds=0)
    persist()
    schemas = action_schemas(env.dual)
    brain.storage.joint_names, brain.storage.cameras = env.joint_names, env.camera_names
    prompt = obs.prompt
    def ruler_reader(photo, side, x_mm, y_mm, step_mm, plane_mm):
        return sample_marks(photo, env.data.camera("front"), env.data.xpos[env.arms[side].base], x_mm, y_mm, step_mm, plane_mm,
                            include_crop=(task == 'stack'))

    brain.begin_episode(raw, prompt, schemas, episode_name=ep, attempt=entry["attempt"], ruler_reader=ruler_reader, mark_spans=(task == 'stack'))
    brain.storage.ensure_frame(obs)
    save_json(raw / "reset.json", dict(seed=seed, frame=obs.frame_id, judge=initial))
    recorder = EpisodeRecorder(raw, brain.storage, env.joint_names, env.camera_names)
    def tick(before, target, after, diagnostics):
        recorder.tick(before, target, after, diagnostics)
        if display and display.poll() is not None:
            raise KeyboardInterrupt("GUI closed")

    executor = ArmExecutor(env, obs, tick)
    reason = "play_round_limit"
    try:
        try:
            executor.home()
        finally:
            brain.storage.decision_start_frame = recorder.count
        recorder.home_end_frame = recorder.count
        recorder.phase = 'vista'
        entry.update(home_end_frame=recorder.count, ticks=recorder.count)
        brain.storage.event('home_completed', start=0, end=recorder.count)
        persist()
        base = env.data.xpos[env.arms["right"].base]
        left = env.data.xpos[env.arms["left"].base] if env.dual else None
        ruler = front_ruler(executor.obs.images["front"], env.data.camera("front"), base, left)
        ruler.save(raw / "front_ruler.png")
        reference = [({"view": "fixed metric ruler, front top plane, mm", "frame_id": executor.obs.frame_id,
                       "top": "right Y", "bottom": "left Y" if env.dual else "right Y", "sides": "X"}, ruler)]
        for round_no in range(1, max_play_rounds + 1):
            if display and display.poll() is not None:
                raise KeyboardInterrupt("GUI closed")
            brain.decide(executor.obs, prompt, schemas, [], reference_views=reference if round_no == 1 else ())
            call = brain.handoff()
            if call["name"] != "play":
                raise RuntimeError("Expected a validated play during collection")
            brain.record_batch_start(brain.active_batch_id)
            entry["play_rounds"] = round_no
            persist()
            end_reason = execute_batch(brain, executor, call["args"]["actions"], call)
            entry["ticks"] = recorder.count
            persist()
            if end_reason:
                reason = end_reason
                break

        # Fix the physical outcome and publish the training data before optional reflection.
        recorder.close()
        entry.update(status="finished", success=reason == "success", reason=reason, ticks=recorder.count)
        persist()  # A finished physical trial can be exported offline after a crash.
        if entry["success"]:
            try:
                publish(data, task, entry)
            except Exception as error:
                entry["export_error"] = str(error)
        persist()
        if not brain._finished:
            brain.review_only = brain.tools.review_only = True
            brain._append([HumanMessage(content="本局物理执行已结束。请独立比较终帧与抓取前的图像，"
                                       "根据本局任务与同步相机证据判断终态。可查历史；仅将按系统提示操作、经前后帧验证的新技巧写入 guide。疑点留在 working。"
                                       "最后用 finish 写明视觉证据和帧号。")])
            try:
                brain.decide(executor.obs, prompt, schemas, [])
                call = brain.handoff()
                if call["name"] != "finish":
                    raise ValueError("Expected a visual finish in review")
                brain.accept_finish_result(dict(tool_call_id=call["id"], accepted=True))
                entry["visual_verdict"] = call["args"]
            except Exception as error:
                entry["review_error"] = str(error)
                brain.storage.abort_batch()
        brain.finalize_collection(ObservationWindow((executor.obs,)), reason)
        brain.storage.outcome(entry)
    except BaseException as error:
        if entry["status"] == "running":
            entry.update(status="interrupted", reason=str(error), ticks=recorder.count)
        brain.abort_episode(error, entry, ObservationWindow((executor.obs,)))
        raise
    finally:
        recorder.close()
        persist()


def provider_config(model):
    return dict(model=model.model, request_options=getattr(model, "options", {}),
                api="responses" if getattr(model, "responses", False) else "chat_completions")


def publish(data, task, entry):
    seed = entry["seed"]
    raw = attempt_path(data, task, seed, entry["attempt"]) / "raw"
    path = data / task / f"ep_{seed:04d}.h5"
    if path.exists() and entry.get("sha256") and file_hash(path) != entry["sha256"]:
        raise ValueError(f"Existing HDF5 checksum differs: {path}")
    names = DUAL_JOINT_NAMES if task == "sort" else JOINT_NAMES
    cameras = read_manifest(data)["tasks"][task]["cameras"]
    export_demo(raw, path, task, TASKS[task].prompt, entry, names, cameras)
    entry.update(hdf5=path.relative_to(data).as_posix(), sha256=file_hash(path))
    entry.pop("export_error", None)


def initialize(data, tasks, seeds, max_play, provider):
    logs = logs_for(data)
    if data.exists() or logs.exists():
        raise FileExistsError(f"Batch already exists: {data.name}. Use --resume.")
    manifest = new_manifest(data.name, tasks, seeds, max_play, provider)
    with TemporaryDirectory(dir=data.parent) as d, TemporaryDirectory(dir=logs.parent) as v:
        staged_data, staged_logs = Path(d) / "data", Path(v) / "logs"
        for task in tasks:
            (staged_data / task).mkdir(parents=True)
            folder = staged_logs / task
            folder.mkdir(parents=True)
            template = Path(__file__).parent / "brains/vista/prompts" / f"{task}.txt"
            prompt_file = folder / "system_prompt.txt"
            prompt_file.write_text(template.read_text("utf8").replace("{max_play_rounds}", str(max_play)), "utf8")
            manifest["tasks"][task]["prompt_sha256"] = file_hash(prompt_file)
            for name in ("guide.md", "guide_start.md"):
                (folder / name).write_text("暂无经过验证的经验。\n", "utf8")
        write_manifest(staged_data, manifest)
        staged_logs.rename(logs)
        try:
            staged_data.rename(data)
        except BaseException:
            logs.rename(staged_logs)
            raise


def record_job(data, task, seed, operation, expected, model_factory=ChatModel, display=None):
    folder = logs_for(data) / task
    # An orphan worker can outlive its coordinator. Never claim its seed twice.
    with ExitStack() as locks:
        try:
            locks.enter_context(FileLock(folder / f".ep_{seed:04d}.lock", timeout=0))
        except BlockingIOError:
            return "already running"
        manifest = read_manifest(data)
        attempts = manifest["tasks"][task]["episodes"][str(seed)]
        if len(attempts) != expected:
            return "already claimed"
        if attempts and attempts[-1]["status"] == "finished" and operation == "record":
            return "already finished"
        if operation == "export":
            entry = attempts[-1]
            try:
                publish(data, task, entry)
            except Exception as error:
                entry["export_error"] = str(error)
                raise
            finally:
                save_attempt(data, task, entry)
            return "exported from raw"
        if attempts and attempts[-1]["status"] == "running":
            stale = attempts[-1]
            stale.update(status="interrupted", reason="worker stopped before finalization")
            save_attempt(data, task, stale)
        if (data / task / f"ep_{seed:04d}.h5").exists():
            raise FileExistsError(f"Unclaimed demonstration already exists for {task} seed {seed}")
        entry = dict(seed=seed, attempt=expected+1, started=True, started_ns=time.time_ns(),
                     status="running", success=False, ticks=0, play_rounds=0)

        def persist():
            save_attempt(data, task, entry)

        persist()
        model = None
        try:
            model = model_factory()
            if provider_config(model) != manifest["provider"]:
                raise ValueError("Model configuration differs from the batch")
            with TabletopEnv(TASKS[task]) as env:
                brain = VistaBrain(run_dir=folder, system_prompt=(folder / "system_prompt.txt").read_text("utf8"),
                                   settings=VistaSettings.from_env(), model=model)
                collect_episode(env, brain, data, entry, persist, display, manifest["max_play_rounds"])
            if entry.get("export_error"):
                raise RuntimeError(entry["export_error"])
        except BaseException as error:
            if entry["status"] != "finished":
                entry.update(status="interrupted", reason=str(error) or type(error).__name__)
            evidence = attempt_path(data, task, seed, entry["attempt"])
            evidence.mkdir(parents=True, exist_ok=True)
            if not (evidence / "outcome.json").exists():
                save_json(evidence / "outcome.json", entry)
            raise
        finally:
            persist()
            if model is not None:
                model.close()
        return entry["reason"]


def run_batch(args, *, root=Path("."), model_factory=ChatModel):
    data = root.resolve() / "data" / (args.out or args.resume)
    logs = logs_for(data)
    data.parent.mkdir(parents=True, exist_ok=True)
    locks = logs.parent / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    with FileLock(locks / (data.name + ".lock"), timeout=0):
        provider = None

        def current_provider():
            nonlocal provider
            if provider is None:
                client = model_factory()
                try:
                    provider = provider_config(client)
                finally:
                    client.close()
            return provider

        if args.out:
            if data.exists() or logs.exists():
                raise FileExistsError(f"Batch already exists: {data.name}. Use --resume.")
            start = args.seed if args.seed is not None else 0
            initialize(data, args.task, range(start, start + (args.episodes or 1)),
                       args.max_play_rounds or MAX_PLAY_ROUNDS, current_provider())
        if not (data / "manifest.json").is_file() or not logs.is_dir():
            raise FileNotFoundError(f"Missing collection or VISTA directory for {data.name}")
        with FileLock(logs / ".manifest.lock"):
            manifest = read_manifest(data)
            if manifest["scene_version"] != SCENE_VERSION:
                raise ValueError("Scene version differs from the batch")
            if args.add:
                extend(manifest, args.add)
            queue = [(task, seed, op, len(manifest["tasks"][task]["episodes"][str(seed)]))
                     for task, seed, op in jobs(data, manifest, args.fix)]
            if any(op == "record" for _, _, op, _ in queue):
                if current_provider() != manifest["provider"]:
                    raise ValueError("Use the model and request options recorded in manifest.json")
                for task, spec in manifest["tasks"].items():
                    if file_hash(logs / task / "system_prompt.txt") != spec["prompt_sha256"]:
                        raise ValueError(f"Saved {task} system prompt has changed")
            if args.add:
                write_manifest(data, manifest)
        print(f"{data}: {len(queue)} scheduled, {args.threads} worker(s)", flush=True)
        if not queue:
            return data
        errors = []
        viewers = {}
        if args.threads == 1:
            for task, seed, op, count in queue:
                if args.gui and op == "record" and task not in viewers:
                    viewers[task] = launch_viewer(data / task)
                try:
                    result = record_job(data, task, seed, op, count, model_factory, viewers.get(task))
                    print(f"{task} seed {seed}: {result}", flush=True)
                except Exception as error:
                    errors.append(f"{task} seed {seed}: {error}")
                    print(errors[-1], flush=True)
                    if not args.fix:
                        break  # A provider failure leaves later seeds queued for resume.
        else:
            with ProcessPoolExecutor(max_workers=args.threads, mp_context=get_context("spawn")) as pool:
                futures = {pool.submit(record_job, data, *job, model_factory): job[:2] for job in queue}
                for future in as_completed(futures):
                    task, seed = futures[future]
                    try:
                        print(f"{task} seed {seed}: {future.result()}", flush=True)
                    except Exception as error:
                        errors.append(f"{task} seed {seed}: {error}")
                        print(errors[-1], flush=True)
        for task in manifest["tasks"]:
            folder = logs / task
            with FileLock(folder / ".guide.lock"):
                (folder / "guide_end.md").write_bytes((folder / "guide.md").read_bytes())
        if errors:
            raise RuntimeError(f"{len(errors)} job(s) reported errors. See {data/'manifest.json'}")
    return data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=batch_name, metavar="NAME", help="Create a named batch")
    mode.add_argument("--resume", type=batch_name, metavar="NAME", help="Continue an existing batch")
    parser.add_argument("--task", choices=TASKS, nargs="+", help="Tasks for a new batch")
    parser.add_argument("--seed", type=int, help="First seed (default 0)")
    parser.add_argument("--episodes", type=int, help="Seeds per task (default 1)")
    parser.add_argument("--max-play-rounds", type=int, help="Play limit for a new batch (default 20)")
    parser.add_argument("--threads", type=int, default=1, help="Total concurrent episode workers")
    parser.add_argument("--add", type=int, metavar="N", help="Append N new seeds to every task")
    parser.add_argument("--fix", action="store_true", help="Recover interrupted recordings and missing exports")
    parser.add_argument("--gui", action="store_true", help="Show task cameras with a single worker")
    args = parser.parse_args(argv)
    if args.out and (not args.task or args.add is not None or args.fix):
        parser.error("--out requires --task; --add and --fix apply to --resume")
    if args.resume and any(v is not None for v in (args.task, args.seed, args.episodes, args.max_play_rounds)):
        parser.error("--resume inherits tasks, seed plan and play limit from the batch")
    if args.task and len(set(args.task)) != len(args.task):
        parser.error("list each task once")
    if args.seed is not None and args.seed < 0:
        parser.error("seed must be nonnegative")
    if any(v is not None and v < 1 for v in (args.episodes, args.threads, args.max_play_rounds, args.add)):
        parser.error("counts must be positive")
    if args.gui and args.threads != 1:
        parser.error("--gui requires --threads 1")
    return args


if __name__ == "__main__":
    try:
        run_batch(parse_args())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(str(error))
