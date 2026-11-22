"""Immutable control frames, atomic decision records, notes and debug logs."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from langchain_core.messages import BaseMessage, message_to_dict
from PIL import Image
from contracts import FrameObservation, ObservationWindow, CONTROL_HZ, JOINT_NAMES, CAMERAS
from .file_lock import FileLock

from .settings import VistaSettings


class HistoryQueryError(ValueError):
    """A recoverable oversized query, not corrupted stored history."""


class GuideConflict(ValueError):
    def __init__(self, latest):
        super().__init__("共享指南已被其他进程更新；请读取最新版、合并需要保留的经验后重写。")
        self.latest = latest


def _json_safe(value):
    # A malformed tool argument may contain NaN in an injected/test model reply.
    # Preserve that fact in diagnostics without writing invalid JSONL.
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_number": str(value)}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_text(path: Path, text: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".vista-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class VistaStorage:
    def __init__(self, settings: VistaSettings, run_dir: Path):
        self.settings = settings
        self.guide_filename = "guide.md"
        self.run_dir = run_dir.resolve()
        self.root = self.run_dir.parent
        self.joint_names, self.cameras = JOINT_NAMES, CAMERAS
        self.episode_dir: Path | None = None
        self.frames: dict[str, dict] = {}
        self._frame_hashes: dict[str, dict[str, str]] = {}
        self.current_frame_id: str | None = None
        self.batch_count = 0
        self.active_batch_id: int | None = None
        self._records: list[dict] = []
        self._last_owned_frame: int | None = None
        self._metadata: dict = {}

    def begin_episode(self, collection_dir: Path, task_text: str, *, episode_name=None) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with self._guide_lock():
                guide = self.root / self.guide_filename
                if not guide.exists():
                    _atomic_text(guide, "暂无经过验证的经验。\n")
            with FileLock(self.root / ".episodes.lock"):
                if episode_name is None:
                    numbers = [int(match.group(1)) for path in self.root.iterdir()
                               if path.is_dir() and (match := re.fullmatch(r"ep_(\d+)", path.name))]
                    episode_name = f"ep_{max(numbers, default=-1) + 1:04d}"
                if not re.fullmatch(r"ep_\d+", episode_name):
                    raise ValueError("无效的 episode 目录名")
                episode = self.run_dir / episode_name
                episode.mkdir(exist_ok=False)
            self.episode_dir = episode.resolve()
            self.frames_dir = collection_dir.resolve() / "frames"
            self.frames_dir.mkdir()
            (episode / "working.md").write_text(
                "当前目标：根据腕部与旁观图像制定计划。\n当前假设：待观察。\n"
                "下一步：检查初始图。\n预期：待提出动作。\n"
                "未决疑点：暂无。每项记录待验证状态或矛盾、关联帧、阻塞的计划、下一检查和解决标准。\n"
                "已解决疑点与直接证据：暂无。\n相关帧编号：暂无。\n",
                encoding="utf-8")
            for name in ("actions", "calls", "messages"):
                (episode / f"{name}.jsonl").touch(exist_ok=False)
        except OSError:
            raise RuntimeError("MEMORY_IO_ERROR: 无法创建 VISTA 图片、笔记或日志目录") from None
        self.frames = {}
        self._frame_hashes = {}
        self.current_frame_id = None
        self.batch_count = 0
        self.active_batch_id = None
        self._records = []
        self.decision_start_frame = 0
        self._last_owned_frame = None
        self._metadata = {"format_version": 3, "fps": CONTROL_HZ,
                          "state_fields": list(self.joint_names),
                          "collection_dir": str(Path(collection_dir).resolve()), "task": task_text}
        self.read_memory("guide")
        self.read_memory("working")
        self.event("episode_start", collection_dir=str(Path(collection_dir).resolve()), task=task_text)

    def _episode(self) -> Path:
        if self.episode_dir is None:
            raise RuntimeError("VISTA 尚未初始化 episode")
        return self.episode_dir

    def append(self, name: str, record: dict) -> None:
        if name not in ("calls", "messages"):
            raise RuntimeError("VISTA 未知日志类型")
        path = self._episode() / f"{name}.jsonl"
        try:
            if not path.is_file() or not path.resolve().is_relative_to(self._episode()):
                raise OSError("log unavailable")
            line = json.dumps(_json_safe(record), ensure_ascii=False, allow_nan=False)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
        except (OSError, TypeError, ValueError):
            raise RuntimeError(f"STORAGE_IO_ERROR: 无法写入 VISTA {name}.jsonl") from None

    def event(self, event: str, **fields) -> None:
        self.append("calls", {"event": event, **fields})

    def diagnostic(self, event: str, original: BaseException, **fields) -> None:
        """Logging a secondary error must never mask the original failure."""
        try:
            self.event(event, error=str(original), **fields)
        except Exception as error:
            original.add_note(f"VISTA 诊断记录也未能保存：{type(error).__name__}")

    def log_message(self, message: BaseMessage) -> None:
        data = message_to_dict(message)
        provider = data["data"].get("additional_kwargs", {}).get("provider_message", {})
        for key in ("reasoning", "reasoning_content", "reasoning_details"):
            provider.pop(key, None)
        content = data["data"]["content"]
        references = iter(message.additional_kwargs.get("vista_images", []))
        if isinstance(content, list):
            for index, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "image_url":
                    reference = next(references, None)
                    if reference is None:
                        raise RuntimeError("VISTA 图片消息缺少来源记录")
                    content[index] = {"type": "image_reference", **reference}
        self.append("messages", data)

    def ensure_frame(self, obs: FrameObservation) -> dict:
        if not isinstance(obs, FrameObservation):
            raise RuntimeError("Expected a synchronized robot observation")
        frame_id = obs.frame_id
        if frame_id in self.frames:
            saved = self.frame_info(frame_id)
            if (saved["state"] != list(obs.state.values) or saved["tcp_mm"] != obs.tcp_mm
                    or any(not np.array_equal(rgb, np.asarray(self.load_image(frame_id, camera)))
                           for camera, rgb in obs.images.items())):
                raise RuntimeError("An archived observation changed")
            return saved
        if obs.frame_no != len(self.frames):
            raise RuntimeError("Discontinuous control frames")
        views, hashes, created = {}, {}, []
        try:
            for camera, rgb in obs.images.items():
                folder = self.frames_dir / camera
                if folder.resolve() != folder:
                    raise OSError("Image directory moved")
                folder.mkdir(exist_ok=True)
                path = folder / f"{frame_id}.png"
                buffer = io.BytesIO()
                Image.fromarray(rgb).save(buffer, format="PNG")
                contents = buffer.getvalue()
                with path.open("xb") as handle:
                    created.append(path)
                    handle.write(contents)
                    handle.flush()
                    os.fsync(handle.fileno())
                views[camera] = {"path": f"frames/{camera}/{frame_id}.png",
                                 "width": rgb.shape[1], "height": rgb.shape[0]}
                hashes[camera] = hashlib.sha256(contents).hexdigest()
        except (OSError, ValueError):
            for path in created:
                path.unlink(missing_ok=True)
            raise RuntimeError(f"FRAME_UNAVAILABLE: unable to publish both cameras for {frame_id}") from None
        info = {"episode_id": self._episode().name, "frame_id": frame_id, "frame_no": obs.frame_no,
                "views": views, "source_batch_id": self.active_batch_id,
                "state": list(obs.state.values), "observation": dict(zip(self.joint_names, obs.state.values)),
                "tcp_mm": obs.tcp_mm}
        self.frames[frame_id] = info
        self._frame_hashes[frame_id] = hashes
        self.current_frame_id = frame_id
        return deepcopy(info)

    def archive_frames(self, frames) -> list[dict]:
        window = ObservationWindow(tuple(frames))
        return [self.ensure_frame(frame) for frame in window.frames]

    def frame_info(self, frame_id: str) -> dict:
        return deepcopy(self.frames[frame_id])

    def load_image(self, frame_id: str, camera: str) -> Image.Image:
        view = self.frames[frame_id]["views"][camera]
        path = self.frames_dir / camera / f"{frame_id}.png"
        try:
            if path.resolve() != path:
                raise OSError("Image path changed")
            contents = path.read_bytes()
            if hashlib.sha256(contents).hexdigest() != self._frame_hashes[frame_id][camera]:
                raise OSError("Published frame was modified")
            with Image.open(io.BytesIO(contents)) as image:
                if image.mode != "RGB" or image.size != (view["width"], view["height"]):
                    raise OSError("Image metadata mismatch")
                return image.copy()
        except (OSError, ValueError):
            raise RuntimeError(f"FRAME_UNAVAILABLE: {camera}/{frame_id}") from None

    def model_views(self, info):
        shared = {key: value for key, value in info.items() if key != "views"}
        return [({**shared, "camera": name, **view}, self.load_image(info["frame_id"], name))
                for name, view in info["views"].items()]

    def check_observation(self, obs: FrameObservation) -> None:
        if not isinstance(obs, FrameObservation) or obs.frame_id != self.current_frame_id:
            raise RuntimeError("VISTA 当前观察与已接收的批次最终图或实测状态不一致，请先接回真实批次结果")
        self.ensure_frame(obs)

    def _memory_path(self, kind: str) -> Path:
        if kind not in ("guide", "working"):
            raise RuntimeError("VISTA 未知笔记类型")
        parent = self.root if kind == "guide" else self._episode()
        path = parent / (self.guide_filename if kind == "guide" else "working.md")
        if not path.is_file() or not path.resolve().is_relative_to(parent):
            raise RuntimeError(f"MEMORY_IO_ERROR: {kind}.md 丢失或路径已改变")
        return path

    def _guide_lock(self):
        return FileLock(self.root / ".guide.lock")

    def _guide_audit(self, operation, **fields):
        """Called with the guide lock held; one complete JSON line per operation."""
        event = {"time_ns": time.time_ns(), "pid": os.getpid(),
                 "episode": self.episode_dir.name if self.episode_dir else None,
                 "operation": operation, **fields}
        with (self.root / "guide_access.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _read_memory_unlocked(self, kind: str) -> dict:
        try:
            content = self._memory_path(kind).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise RuntimeError(f"MEMORY_IO_ERROR: 无法读取 {kind}.md") from None
        maximum = self.settings.max_guide_chars if kind == "guide" else self.settings.max_working_chars
        if not content.strip() or len(content) > maximum:
            raise RuntimeError(f"MEMORY_IO_ERROR: {kind}.md 为空或超出 {maximum} 字符")
        result = {"content": content, "char_count": len(content),
                  "scope": "environment" if kind == "guide" else "episode"}
        if kind == "guide":
            result["revision"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return result

    def read_memory(self, kind: str, *, reader="tool") -> dict:
        if kind != "guide":
            return self._read_memory_unlocked(kind)
        with self._guide_lock():
            result = self._read_memory_unlocked(kind)
            self._guide_audit("read", revision=result["revision"], reader=reader)
            return result

    def write_memory(self, kind: str, content: str, *, base_revision=None) -> dict:
        # Tool validation has already checked the full replacement text.
        try:
            if kind == "guide":
                with self._guide_lock():
                    current = self._read_memory_unlocked(kind)
                    if base_revision != current["revision"]:
                        self._guide_audit("conflict", base_revision=base_revision,
                                          current_revision=current["revision"])
                        raise GuideConflict(current)
                    revision = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    archive = self.root / "guide_versions"
                    archive.mkdir(exist_ok=True)
                    for version, text in ((current["revision"], current["content"]), (revision, content)):
                        if not (archive / f"{version}.md").exists():
                            _atomic_text(archive / f"{version}.md", text)
                    self._guide_audit("write_attempt", base_revision=base_revision, revision=revision)
                    _atomic_text(self._memory_path(kind), content)
                    self._guide_audit("write", base_revision=base_revision, revision=revision)
                    return {**self._read_memory_unlocked(kind), "written": True}
            _atomic_text(self._memory_path(kind), content)
        except (OSError, UnicodeError):
            raise RuntimeError(f"MEMORY_IO_ERROR: 无法完整写入 {kind}.md") from None
        return {**self.read_memory(kind), "written": True}

    def _commit_records(self, records: list[dict], active_batch_id: int | None) -> None:
        """Publish disk first; a failed replacement must not advance memory state."""
        self._validate_records(records)
        path = self._episode() / "actions.jsonl"
        try:
            if not path.is_file() or not path.resolve().is_relative_to(self._episode()):
                raise OSError("actions unavailable")
            _atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                      for row in records))
        except (OSError, UnicodeError, TypeError, ValueError):
            raise RuntimeError("STORAGE_IO_ERROR: 无法原子更新 VISTA actions.jsonl") from None
        self._records = records
        self.batch_count = len(records)
        self.active_batch_id = active_batch_id
        owned = [obs["frame_id"] for record in records for obs in record["observations"]]
        self._last_owned_frame = int(owned[-1][1:]) if owned else None

    def _validate_records(self, records: list[dict]) -> None:
        expected_frame = previous_end = self.decision_start_frame
        for index, record in enumerate(records, 1):
            required = {"batch_id", "tool_call_id", "frame_range", "actions", "observations", "execution_status"}
            if not required <= set(record) or set(record) - required - {"observation_error"}:
                raise ValueError("invalid record fields")
            start, end = record["frame_range"]
            if (record["batch_id"] != index or type(start) is not int or type(end) is not int
                    or start != previous_end or end < start
                    or not isinstance(record["actions"], list)
                    or record["execution_status"] not in (
                        "pending", "completed", "partial", "failed", "unknown", "cancelled", "finished", "interrupted")):
                raise ValueError("invalid decision record")
            observed = record["observations"]
            incomplete = record.get("observation_error") is not None
            if (end > start and (len(observed) > end - start if incomplete else len(observed) != end - start)
                    or end == start and len(observed) not in (0, 1)):
                raise ValueError("observation count does not match frame range")
            for offset, obs in enumerate(observed):
                frame = self.frames[f"f{expected_frame:06d}"]
                if (start + offset != expected_frame
                        or obs != {"frame_id": frame["frame_id"], "state": frame["state"]}):
                    raise ValueError("observations are repeated, missing or differ from archived state")
                expected_frame += 1
            previous_end = end

    def _active_record(self, batch_id: int) -> dict:
        if type(batch_id) is not int or batch_id != self.active_batch_id:
            raise RuntimeError("VISTA 批次编号不匹配或已经结算")
        return self._records[-1]

    def observation_record(self, frame_no: int) -> dict:
        info = self.frame_info(f"f{frame_no:06d}")
        return {"frame_id": info["frame_id"], "state": info["state"]}

    def begin_batch(self, start_frame: int) -> int:
        if self.active_batch_id is not None:
            raise RuntimeError("VISTA 上一批次尚未结算")
        if self.current_frame_id != f"f{start_frame:06d}":
            raise RuntimeError("VISTA 新批次必须从当前帧开始")
        records = deepcopy(self._records)
        if self._last_owned_frame == start_frame:
            # A zero-tick failure may be followed by a recovery decision on the same
            # frame. Move ownership atomically instead of duplicating this snapshot.
            if not records or records[-1]["frame_range"] != [start_frame, start_frame]:
                raise RuntimeError("VISTA 起始观测已被其他非空区间占用")
            records[-1]["observations"] = []
        batch_id = len(records) + 1
        records.append({"batch_id": batch_id, "tool_call_id": None,
                        "frame_range": [start_frame, start_frame], "actions": [],
                        "observations": [self.observation_record(start_frame)], "execution_status": "pending"})
        self._commit_records(records, batch_id)
        return batch_id

    def bind_batch_call(self, batch_id: int, tool_call_id: str, actions: list[dict]) -> None:
        record = self._active_record(batch_id)
        if record["tool_call_id"] is not None or not isinstance(tool_call_id, str) or not tool_call_id:
            raise RuntimeError("VISTA 交接调用重复绑定或缺少 ID")
        records = deepcopy(self._records)
        records[-1]["tool_call_id"] = tool_call_id
        records[-1]["actions"] = deepcopy(actions)
        self._commit_records(records, batch_id)

    def validate_batch_window(self, batch_id: int, window: ObservationWindow) -> None:
        record = self._active_record(batch_id)
        if not isinstance(window, ObservationWindow) or window.frames[0].frame_no != record["frame_range"][0]:
            raise RuntimeError("OBSERVATION_UNAVAILABLE: 批次观测窗口起点不匹配")
        self.ensure_frame(window.frames[0])

    def complete_batch(self, batch_id: int, end_frame: int, action_results: list[dict],
                       execution_status: str, observation_error: dict | None = None) -> None:
        record = self._active_record(batch_id)
        start = record["frame_range"][0]
        saved_end = self.frames[self.current_frame_id]["frame_no"]
        if end_frame < start or (observation_error is None and end_frame != saved_end):
            raise RuntimeError("VISTA 批次终点不是当前已归档帧")
        if len(action_results) != len(record["actions"]):
            raise RuntimeError("VISTA 批次必须结算所有请求动作")
        records = deepcopy(self._records)
        updated = records[-1]
        updated["observations"] = [self.observation_record(i)
                                   for i in range(start, min(max(start + 1, end_frame), saved_end + 1))]
        updated["frame_range"] = [start, end_frame]
        updated["actions"] = deepcopy(action_results)
        updated["execution_status"] = execution_status
        if observation_error is not None:
            updated["observation_error"] = deepcopy(observation_error)
        self._commit_records(records, None)

    def finish_batch(self, batch_id: int) -> None:
        if self._active_record(batch_id)["actions"]:
            raise RuntimeError("VISTA finish 不能结算有待执行动作的批次")
        records = deepcopy(self._records)
        records[-1]["execution_status"] = "finished"
        self._commit_records(records, None)

    def ensure_terminal_observation(self, stop_code: str = "finished") -> None:
        if self.active_batch_id is not None:
            raise RuntimeError("VISTA 结束前必须结算当前批次")
        if self.current_frame_id is None:
            return
        end = self.frames[self.current_frame_id]["frame_no"]
        if self._last_owned_frame == end:
            return
        batch_id = self.begin_batch(end)
        records = deepcopy(self._records)
        records[-1]["execution_status"] = "interrupted" if stop_code == "interrupted" else "finished"
        self._commit_records(records, None)

    def abort_batch(self) -> None:
        if self.active_batch_id is None:
            return
        record = self._records[-1]
        results = deepcopy(record["actions"])
        for index, result in enumerate(results, 1):
            result.update(index=index, status="unknown", action_applied=None, frame_range=None,
                          error={"code": "EXECUTION_INTERRUPTED", "message": "中断前未收到可靠执行回执"})
        end = self.frames[self.current_frame_id]["frame_no"]
        self.complete_batch(self.active_batch_id, end, results, "interrupted")

    def recover_observations(self) -> None:
        """Attach successfully retried PNGs to a failed collection, keeping action facts."""
        if not self._records or "observation_error" not in self._records[-1]:
            return
        record = self._records[-1]
        start, end = record["frame_range"]
        saved_end = self.frames[self.current_frame_id]["frame_no"]
        if saved_end > end:
            raise RuntimeError("VISTA 恢复观测超过已知执行范围")
        records = deepcopy(self._records)
        records[-1]["observations"] = [self.observation_record(i)
                                       for i in range(start, min(max(start + 1, end), saved_end + 1))]
        if saved_end == end:
            records[-1]["observation_error"]["recovered"] = True
        self._commit_records(records, self.active_batch_id)

    def history(self, start=None, end=None, limit=None, *, mode="batch", step=1) -> dict:
        """Read complete batches or sampled frames; ranges include both ends.

        Explicit starts page forward. Without a start, show the latest page;
        next_query then points to the older omitted range, without losing frames.
        """
        limit = self.settings.history_default_limit if limit is None else limit
        minimum = 1 if mode == "batch" else 0
        if (mode not in ("batch", "frame")
                or any(value is not None and (type(value) is not int or value < minimum)
                       for value in (start, end))
                or (start is not None and end is not None and start > end)
                or type(step) is not int or step < 1 or (mode == "batch" and step != 1)
                or type(limit) is not int or not 1 <= limit <= self.settings.history_max_limit):
            raise ValueError("invalid history query")
        try:
            path = self._episode() / "actions.jsonl"
            if not path.resolve().is_relative_to(self._episode()):
                raise ValueError("history path changed")
            entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if entries != self._records:
                raise ValueError("history changed outside storage")
            self._validate_records(entries)
        except (OSError, UnicodeError, ValueError, TypeError, KeyError):
            raise RuntimeError("HISTORY_UNAVAILABLE: 批次历史丢失或损坏") from None
        current = int(self.current_frame_id[1:]) if self.current_frame_id is not None else None
        last = len(entries) if mode == "batch" else current
        lower = minimum if start is None else start
        upper = (last if end is None else min(end, last)) if last is not None else -1
        numbers = range(lower, upper + 1, step)
        selected = numbers[-limit:] if start is None else numbers[:limit]
        truncated = len(selected) < len(numbers)
        next_query = None
        if truncated:
            next_query = {"mode": mode, "start": lower if start is None else selected[-1] + step,
                          "end": selected[0] - step if start is None else upper, "limit": limit}
            if mode == "frame":
                next_query["step"] = step
        result = {"mode": mode, "current_frame_no": current,
                  "available_frame_range": [0, current] if current is not None else None,
                  "available_batch_range": [1, len(entries)] if entries else None,
                  "state_fields": list(self.joint_names), "truncated": truncated,
                  "returned_range": [selected[0], selected[-1]] if selected else None,
                  "next_query": next_query}
        if mode == "batch":
            batches = [entries[number - 1] for number in selected]
            count = sum(len(row["observations"]) for row in batches)
            if count > self.settings.history_max_observations:
                raise HistoryQueryError(
                    f"所选批次包含 {count} 个观测，单次上限 {self.settings.history_max_observations}；"
                    "请缩小批次范围或改用 mode=frame。"
                    f"所选批次帧范围为 [{batches[0]['frame_range'][0]}, {batches[-1]['frame_range'][1]}]；"
                    "frame 查询可用 step 抽样、limit 分页，批次记录不会被拆开或省略观测。")
            result["batches"] = batches
        else:
            # Ownership follows observations, not the batch that produced a frame.
            # A just-accepted boundary has no owner until the next decide/finish.
            owners = {obs["frame_id"]: row["batch_id"] for row in entries for obs in row["observations"]}
            result["frames"] = [
                {"frame_no": number, **self.observation_record(number),
                 "batch_id": owners.get(f"f{number:06d}")} for number in selected]
        return result

    def outcome(self, outcome: dict) -> None:
        try:
            _atomic_text(self._episode() / "outcome.json",
                         json.dumps({**outcome, **self._metadata}, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        except (OSError, ValueError, TypeError):
            raise RuntimeError("STORAGE_IO_ERROR: 无法保存 VISTA episode 结局") from None
