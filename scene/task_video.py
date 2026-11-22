"""Side-by-side sensor recording, with task diagnostics confined to the footer."""
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from contracts import CONTROL_HZ


def footer(info):
    if "cube_clearance_m" in info:
        return (f"red bottom {1000 * info['cube_clearance_m']:.1f} mm   "
                f"fixed={int(info['fixed_contact'])} moving={int(info['moving_contact'])}   success={int(info['success'])}")
    if "inside" in info:
        detail = "in zone " + " ".join(f"{c}={int(v)}" for c, v in info["inside"].items())
    else:
        detail = f"center error {info['center_error_mm']:.1f} mm   supported={int(info['red_supported'])}"
    return f"{detail}   released={int(info['released'])}   stable={info['stable_ticks']}   success={int(info['success'])}"


class TaskVideo:
    def __init__(self, path, cameras):
        self.cameras = tuple(cameras)
        self.writer = imageio_ffmpeg.write_frames(str(path), (480*len(cameras), 432), fps=CONTROL_HZ,
                                                 codec="libx264", macro_block_size=1)
        self.writer.send(None)
        self.font = ImageFont.load_default(size=16)

    def add(self, images, label, info):
        canvas = Image.new("RGB", (480*len(self.cameras), 432), "#182028")
        draw = ImageDraw.Draw(canvas)
        for i, c in enumerate(self.cameras):
            frame = images[c]
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(frame)
            canvas.paste(frame.resize((480, 360), Image.Resampling.BILINEAR), (480*i, 0))
            draw.text((480*i+8, 364), c, font=self.font, fill="#b9d4e7")
        draw.text((8, 387), label, font=self.font, fill="white")
        draw.text((8, 409), footer(info), font=self.font, fill="white")
        self.writer.send(np.asarray(canvas))
        return canvas

    def close(self):
        self.writer.close()


class Preview:
    def __init__(self, title):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.title(title)
        self.panel = tk.Label(self.root)
        self.panel.pack()

    def show(self, frame):
        from PIL import ImageTk
        frame = frame.copy()
        frame.thumbnail((self.root.winfo_screenwidth()-50, self.root.winfo_screenheight()-130))
        self.photo = ImageTk.PhotoImage(frame)
        self.panel.configure(image=self.photo)
        self.root.update()

    def close(self):
        self.root.destroy()
