"""A metric ruler on a second copy of the front RGB, for manual VISTA inspection."""
from itertools import groupby

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scene.camera import FRONT_D, FRONT_K, radial_distance


def project_marks(camera, base, xs, ys, plane_mm=30):
    xyz = np.stack((base[0]-ys/1000, base[1]+xs/1000,
                    np.full_like(xs,plane_mm/1000,dtype=float)), axis=-1)
    ray = (xyz-camera.xpos) @ camera.xmat.reshape(3,3)
    x, y = -ray[...,0]/ray[...,2], ray[...,1]/ray[...,2]
    _, _, p1, p2, _ = FRONT_D
    r2 = x*x+y*y
    radius = np.sqrt(r2)
    radial = np.divide(radial_distance(radius), radius, out=np.ones_like(radius), where=radius>0)
    u = FRONT_K[0,0]*(x*radial+2*p1*x*y+p2*(r2+2*x*x))+FRONT_K[0,2]
    v = FRONT_K[1,1]*(y*radial+p1*(r2+2*y*y)+2*p2*x*y)+FRONT_K[1,2]
    return u, v


def sample_marks(rgb, camera, base, x_mm, y_mm, step_mm, plane_mm=30, *, include_crop=False):
    """Nine by nine model-chosen fixed marks. Return original sensor RGB, in scan order."""
    offsets = np.arange(-4,5)*step_mm
    xs, ys = np.meshgrid(x_mm+offsets, y_mm+offsets, indexing="ij")
    u, v = project_marks(camera, base, xs, ys, plane_mm)
    photo = np.asarray(rgb)
    samples = []
    for i in range(len(offsets)):
        cells = []
        for j in range(len(offsets)):
            col, row = round(float(u[i,j])), round(float(v[i,j]))
            value = bytes(photo[row,col]).hex() if 0<=row<photo.shape[0] and 0<=col<photo.shape[1] else "outside"
            cells.append(f"Y{ys[i,j]:+d}=#{value}")
        samples.append(f"X{xs[i,0]} | " + " ".join(cells))
    response = {"samples_rgb":samples}
    if not include_crop:
        return response, None
    if step_mm <= 5:
        # Both traces report every colour along a fixed ruler. Group equal nibbles
        # for readability; no object is selected or measured automatically.
        lines = {}
        for axis in ("X", "Y"):
            offset = np.arange(-30,31)
            xx = x_mm+(offset if axis=="X" else 0*offset)
            yy = y_mm+(offset if axis=="Y" else 0*offset)
            u, v = project_marks(camera,base,xx,yy,plane_mm)
            points = []
            for mark,col,row in zip(xx if axis=="X" else yy,np.rint(u).astype(int),np.rint(v).astype(int)):
                tone = "".join(format(int(ch)>>4,"x") for ch in photo[row,col]) if 0<=row<photo.shape[0] and 0<=col<photo.shape[1] else "outside"
                points.append((int(mark),tone))
            spans = []
            for tone, values in groupby(points,key=lambda point:point[1]):
                values = list(values)
                spans.append(f"{values[0][0]}..{values[-1][0]}:#{tone}")
            lines[axis+"_mm"] = " ".join(spans)
        response["crosshair_lines_rgb4bit"] = lines
    # Crosshair at the model's proposed position, with conventional Cartesian axes.
    pixels = np.linspace(-4*step_mm,4*step_mm,321)
    xx, yy = np.meshgrid(x_mm+pixels, y_mm-pixels)
    u, v = project_marks(camera, base, xx, yy, plane_mm)
    photo = Image.fromarray(cv2.remap(photo,u.astype("f4"),v.astype("f4"),cv2.INTER_LINEAR))
    draw = ImageDraw.Draw(photo)
    for i in range(0,321,40):
        draw.line((i,156,i,164),fill="#ffffff",width=2)
        draw.line((156,i,164,i),fill="#ffffff",width=2)
    for line in ((137,160,152,160),(168,160,183,160),(160,137,160,152),(160,168,160,183)):
        draw.line(line,fill="#202020",width=5)
        draw.line(line,fill="#00e5ff",width=2)
    canvas = Image.new("RGB",(321,356),"#fcfcfc")
    canvas.paste(photo)
    ImageDraw.Draw(canvas).text((5,329),f"center ({x_mm},{y_mm:+d})    +X right    +Y up    ticks {step_mm} mm",
                               fill="black",font=ImageFont.load_default(size=14))
    return response, canvas


def front_ruler(rgb, camera, base, left_base=None):
    """Resample the 30 mm plane using only fixed camera and robot-base calibration."""
    dual = left_base is not None
    y0, y1 = (90, 260) if dual else (-90, 90)
    x0, x1, scale = 135, 310, 4
    cols, rows = np.meshgrid(np.arange((y1-y0)*scale+1), np.arange((x1-x0)*scale+1))
    u, v = project_marks(camera, base, x0+rows/scale, y0+cols/scale)
    surface = Image.fromarray(cv2.remap(rgb,u.astype("f4"),v.astype("f4"),cv2.INTER_LINEAR))
    gx, gy = 60, 58
    result = Image.new("RGB",(surface.width+120,surface.height+126),"#fafafa")
    result.paste(surface,(gx,gy))
    pen = ImageDraw.Draw(result)
    font = ImageFont.load_default(size=19)
    small = ImageFont.load_default(size=14)
    width, height = surface.size
    for yval in range(y0, y1+1, 5):
        u = gx+(yval-y0)*scale
        for xval in range(x0, x1+1, 5):
            v = gy+(xval-x0)*scale
            # Sparse registration dots leave cube edges and letters visible.
            pen.point((u,v), fill="#416585")
            if yval%20==0 and xval%20==0:
                pen.line((u-3,v,u+3,v),fill="#457da5")
                pen.line((u,v-3,u,v+3),fill="#457da5")
        if yval%10==0:
            pen.line((u,gy-6,u,gy+height),fill="#7997ab",width=1)
            if yval%20==0:
                pen.text((u,gy-18),f"{yval:+d}",font=font,anchor="mm",fill="#15314f")
                lower = yval if not dual else yval + round((left_base[0]-base[0])*1000)
                pen.text((u,gy+height+18),f"{lower:+d}",font=font,anchor="mm",fill="#15314f")
    for xval in range(140,x1+1,10):
        v=gy+(xval-x0)*scale
        pen.line((gx-5,v,gx+width,v),fill="#7997ab")
        if xval%20==0:
            for u in (gx-29,gx+width+29):
                pen.text((u,v),str(xval),font=font,anchor="mm",fill="#15314f")
    pen.text((gx,4),"TOP: right arm Y (mm)     SIDES: X (mm), increasing down",font=small,fill="black")
    bottom = "left arm Y" if dual else "right arm Y"
    pen.text((gx,gy+height+36),"BOTTOM: "+bottom+" (mm)     Lines every 10 mm; dots every 5 mm.",
             font=small,fill="black")
    return result
