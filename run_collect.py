"""Collect a seeded batch of tabletop demonstrations with VISTA."""
import argparse
import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from brains.vista.brain import VistaBrain
from brains.vista.model import ChatModel
from brains.vista.settings import VistaSettings
from brains.vista.state import action_result, batch_status
from collection.recorder import EpisodeRecorder, save_json
from contracts import DUAL_CAMERAS, CAMERAS, ObservationWindow, SCENE_VERSION
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


def collect_episode(env, brain, data_dir, entry, persist, source, display=None, *, max_play_rounds=MAX_PLAY_ROUNDS):
    seed = entry["seed"]
    ep = f"ep_{seed:04d}"
    raw = data_dir / "raw" / ep
    raw.mkdir(parents=True)
    obs, initial = env.reset(seed=seed, finish_on_success=False, limit_ticks=False)
    entry.update(status="running", started=True, success=False, ticks=0, play_rounds=0)
    persist()
    task = type(env.task).__name__.lower()
    schemas = action_schemas(env.dual)
    brain.storage.joint_names, brain.storage.cameras = env.joint_names, env.camera_names
    prompt = obs.prompt
    def ruler_reader(photo, side, x_mm, y_mm, step_mm, plane_mm):
        return sample_marks(photo, env.data.camera("front"), env.data.xpos[env.arms[side].base], x_mm, y_mm, step_mm, plane_mm,
                            include_crop=(task == 'stack'))

    brain.begin_episode(raw, prompt, schemas, episode_name=ep, ruler_reader=ruler_reader, mark_spans=(task == 'stack'))
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
        if entry["success"]:
            h5 = data_dir / "episodes" / (ep + ".h5")
            try:
                recorder.export(h5, task, prompt, seed, source)
                entry["hdf5"] = h5.relative_to(data_dir).as_posix()
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


def create_run(root):
    root.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        path = root / f"run{i:02d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            i += 1


def run(seed, episodes, model_factory=ChatModel, *, root=Path("."), source="vista", gui=False,
        max_play_rounds=MAX_PLAY_ROUNDS, task="lift"):
    if type(max_play_rounds) is not int or max_play_rounds < 1:
        raise ValueError("max_play_rounds must be a positive integer")
    settings = VistaSettings.from_env()
    model = model_factory()
    data = create_run(root / "data" / task)
    (data / "episodes").mkdir()
    memory = root / "output/vista" / task
    vista_run = memory / data.name
    vista_run.mkdir(parents=True)
    guide = memory / "guide_track1.md"
    if not guide.exists():
        with guide.open("x", encoding="utf8") as f:
            f.write("暂无经过验证的经验。\n")
    template = (Path(__file__).parent / "brains/vista/prompts" / f"{task}.txt").read_text("utf8")
    system_prompt = template.replace("{max_play_rounds}", str(max_play_rounds))
    (vista_run / "system_prompt.txt").write_text(system_prompt, encoding="utf8")
    (vista_run / "guide_start.md").write_bytes(guide.read_bytes())
    entries = [dict(seed=n, status="planned", started=False) for n in range(seed, seed + episodes)]
    manifest = dict(scene_version=SCENE_VERSION, task=task, source=source, model=model.model, cameras=list(DUAL_CAMERAS if task == "sort" else CAMERAS),
                    max_play_rounds=max_play_rounds, episodes=entries)

    if hasattr(model, "options"):
        manifest["request_options"] = dict(model.options)

    def persist():
        started = sum(e["started"] for e in entries)
        succeeded = sum(e.get("success", False) for e in entries)
        manifest["summary"] = dict(planned=len(entries), started=started, success=succeeded,
            interrupted=sum(e["status"] == "interrupted" for e in entries),
            exported=sum("hdf5" in e for e in entries), success_rate=succeeded / started if started else None)
        save_json(data / "manifest.json", manifest)

    persist()
    print(f"{data}  model={model.model}  seeds={seed}..{seed + episodes - 1}", flush=True)
    try:
        with TabletopEnv(TASKS[task]) as env:
            display = launch_viewer(data) if gui else None
            for entry in entries:
                brain = VistaBrain(run_dir=vista_run, system_prompt=system_prompt, settings=settings, model=model)
                brain.storage.guide_filename = guide.name
                try:
                    collect_episode(env, brain, data, entry, persist, source, display,
                                    max_play_rounds=max_play_rounds)
                except Exception as error:
                    if entry["status"] != "finished":
                        entry.update(status="interrupted", reason=str(error))
                    persist()
                    print(f"seed {entry['seed']}: {error}", flush=True)
                    # Systemic API/configuration failures should not consume the rest of a batch.
                    raise
                print(f"seed {entry['seed']}: {entry['reason']}, {entry['ticks']} ticks", flush=True)
    finally:
        (vista_run / "guide_end.md").write_bytes(guide.read_bytes())
        model.close()
        persist()
    print(json.dumps(manifest["summary"]), flush=True)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-play-rounds", type=int, default=MAX_PLAY_ROUNDS,
                        help="Maximum play calls per episode (default: 20)")
    parser.add_argument("--gui", action="store_true", help="Show all synchronized task cameras")
    args = parser.parse_args()
    if args.seed < 0 or args.episodes < 1 or args.max_play_rounds < 1:
        parser.error("seed must be nonnegative; episodes and max-play-rounds must be positive")
    run(args.seed, args.episodes, gui=args.gui, task=args.task, max_play_rounds=args.max_play_rounds)
