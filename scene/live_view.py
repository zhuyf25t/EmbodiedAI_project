"""Show the archived sensor frames in a GUI that stays responsive during model calls."""
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image

from collection.batch import attempt_path, logs_for, read_manifest
from contracts import CONTROL_HZ


def launch_viewer(run_dir):
    python = Path(sys.executable)
    if os.name == "nt" and python.with_name("pythonw.exe").exists():
        python = python.with_name("pythonw.exe")
    with (logs_for(run_dir.parent) / run_dir.name / "viewer.log").open("a", encoding="utf8") as log:
        return subprocess.Popen([str(python), "-m", "scene.live_view", str(run_dir.resolve())],
                                cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=log)


def new_lines(path, offset):
    """Consume complete records only. A writer can be halfway through a JSON line."""
    if not path.exists():
        return [], offset
    records = []
    with path.open("rb") as stream:
        stream.seek(offset)
        for line in stream:
            if not line.endswith(b"\n"):
                break
            records.append(json.loads(line))
            offset += len(line)
    return records, offset


def describe(action):
    if action["action"] == "move_by":
        args = action["args"]
        return args.get("arm", "") + " 移动  " + "  ".join(f"{k[1].upper()} {args[k]:+g}" for k in ("dx_mm", "dy_mm", "dz_mm")) + " mm"
    return action.get("args", {}).get("arm", "") + " " + {"open_gripper": "张爪", "close_gripper": "闭爪", "observe": "保持目标"}.get(action["action"], action["action"])


class RunWindow:
    def __init__(self, root, directory):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.directory = root, directory.resolve()
        self.data, self.task = self.directory.parent, self.directory.name
        self.cameras = read_manifest(self.data)["tasks"][self.task]["cameras"]
        root.title(f"VISTA · {self.task.title()} · {self.data.name}")
        self.width = min(640, max(240, (root.winfo_screenwidth() - 80) // len(self.cameras)),
                         max(240, (root.winfo_screenheight() - 240) * 4 // 3))
        box = ttk.Frame(root, padding=8)
        box.pack()
        self.panels = []
        for i, label in enumerate(self.cameras):
            ttk.Label(box, text=label).grid(row=0, column=i, sticky="w")
            panel = tk.Canvas(box, width=self.width, height=self.width * 3 // 4,
                              background="#20272c", highlightthickness=0)
            panel.grid(row=1, column=i)
            panel.create_image(0, 0, anchor="nw", tags="rgb")
            self.panels.append(panel)
        self.caption = tk.StringVar(value="正在等待第一帧…")
        ttk.Label(box, textvariable=self.caption).grid(row=2, columnspan=len(self.cameras), sticky="w", pady=6)
        self.activity = tk.StringVar()
        ttk.Label(box, textvariable=self.activity).grid(row=3, columnspan=len(self.cameras), sticky="w")
        self.replay = ttk.Button(box, text="从头回放", command=self.restart)
        self.replay.grid(row=4, column=0, sticky="w", pady=6)
        self.episode = None
        self.current = -1
        self.root.after(0, self.refresh)

    def restart(self):
        self.current = -1

    def select_episode(self, seed, attempt):
        self.episode = seed, attempt
        folder = attempt_path(self.data, self.task, seed, attempt)
        self.raw, self.log = folder / "raw", folder / "calls.jsonl"
        self.tick_offset = self.call_offset = self.latest = 0
        self.current = -1
        self.batch = 0
        self.actions = []
        self.boundaries = {}
        self.last_tool = "初始化"
        self.active_actions = []
        self.start = 0

    def read_activity(self):
        events, self.call_offset = new_lines(self.log, self.call_offset)
        for event in events:
            kind = event["event"]
            if kind == "tool_request":
                self.last_tool = event["name"]
            elif kind == "model_request":
                self.activity.set(f"等待 VISTA 回复  ·  最近工具 {self.last_tool}")
            elif kind == "batch_start":
                self.batch = event["batch_id"]
                self.active_actions = list(event["args"]["actions"])
                self.activity.set(f"执行第 {self.batch} 轮动作")
            elif kind == "control_boundary":
                end = event["end"]
                if self.active_actions:
                    self.actions.append((event["start"], end, self.batch, describe(self.active_actions.pop(0))))
                self.start = end
                if event["error"] or event["stable_frames"]:
                    self.boundaries[end] = event["error"] or "连续两帧稳定"
                if event["error"]:
                    self.active_actions.clear()

    def show_frame(self, index):
        from PIL import ImageTk

        photos = []
        for camera in self.cameras:
            with Image.open(self.raw / "frames" / camera / f"f{index:06d}.png") as im:
                rgb = im.resize((self.width, self.width * 3 // 4), Image.Resampling.BILINEAR)
                photos.append(ImageTk.PhotoImage(rgb, master=self.root))
        for panel, photo in zip(self.panels, photos):
            panel.itemconfigure("rgb", image=photo)
            panel.photo = photo
        self.current = index

    def refresh(self):
        try:
            self.update()
        except (FileNotFoundError, PermissionError, OSError, json.JSONDecodeError):
            # The first pair and the manifest may still be in the process of publication.
            pass
        self.root.after(round(1000 / CONTROL_HZ), self.refresh)

    def update(self):
        task = read_manifest(self.data)["tasks"][self.task]
        latest = [row[-1] for row in task["episodes"].values() if row]
        if not latest:
            return
        entry = max(latest, key=lambda e: e["started_ns"])
        key = entry["seed"], entry["attempt"]
        if self.episode != key:
            self.select_episode(*key)
        if not (self.raw / "reset.json").is_file():
            return
        ticks, self.tick_offset = new_lines(self.raw / "ticks.jsonl", self.tick_offset)
        if ticks:
            self.latest = int(ticks[-1]["next_frame"][1:])
            for t in ticks:
                if t.get("phase") == "home":
                    self.actions.append((int(t["frame"][1:]), int(t["next_frame"][1:]), 0, "初姿回位"))
        self.read_activity()
        if self.current < self.latest:
            self.show_frame(self.current + 1)
        if self.current < 0:
            return
        number, action = 0, "初始布局"
        for start, end, n, label in reversed(self.actions):
            if start < self.current <= end:
                number, action = n, label
                break
        if self.active_actions and self.current > self.start:
            number, action = self.batch, describe(self.active_actions[0])
        boundary = self.boundaries.get(self.current, "")
        self.caption.set(f"seed {entry['seed']}    f{self.current:06d} / f{self.latest:06d}    "
                         f"第 {number} 轮    {action}    {boundary}")
        if entry['status'] in ('finished', 'interrupted'):
            reason = {"success": "任务成功", "play_round_limit": "达到动作轮数上限",
                      "settle_timeout": "保持超时"}.get(entry.get("reason"), entry.get("reason", ""))
            self.activity.set(f"采集已结束  ·  {reason}  ·  {entry.get('play_rounds', self.batch)} 轮"
                              f"  ·  {entry['ticks']} 控制拍")


if __name__ == "__main__":
    import argparse
    import tkinter as tk

    parser = argparse.ArgumentParser(description="Watch or replay all cameras from a VISTA run")
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = tk.Tk()
    window = RunWindow(root, args.run)
    root.mainloop()
