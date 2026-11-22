"""Audit seeded starts and IK paths, without a policy or a model API call."""
import json
from pathlib import Path

import mujoco
import numpy as np

from control.vista import ArmExecutor
from PIL import Image, ImageDraw

from scene.env import TabletopEnv
from scene.layout import square_corners, overlap
from scene.lift import Lift
from scene.stack import Stack
from scene.sort import Sort


def inspect(task):
    starts, probes = [], []
    with TabletopEnv(task) as env:
        for seed in range(16):
            obs, _ = env.reset(seed=seed)
            cell = env.data.site("mixed" if env.dual else "red_target")
            half = env.model.site_size[cell.id,:2]
            blocks, corners = {}, []
            for color, geom in env.task.cubes.items():
                pos = env.data.geom_xpos[geom,:2]
                rot = env.data.geom_xmat[geom].reshape(3,3)
                angle = np.arctan2(rot[1,0], rot[0,0])
                box = square_corners(pos, angle)
                assert np.all(abs(box-cell.xpos[:2]) <= half + 1e-5)
                blocks[color] = [*pos.tolist(), float(np.rad2deg(angle))]
                corners.append(box)
            if len(corners) == 2:
                assert not overlap(*corners)
            starts.append(dict(seed=seed, arms=obs.state.values, blocks=blocks))
        for side in env.arms:
            cell = env.data.site("mixed" if env.dual else "red_target")
            dest = None
            if task is Stack or env.dual:
                dest = env.data.site("red_target" if side=="right" else "green_target").xpos[:2]
            for angle in (0,15,30,45):
                half = env.model.site_size[cell.id,:2] - .015*(np.cos(np.deg2rad(angle))+np.sin(np.deg2rad(angle))) - .001
                for a in (-1,0,1):
                    for b in (-1,0,1):
                        pos = cell.xpos[:2] + half * [a,b]
                        ok = env.layout.path(side, pos, angle, dest)
                        probes.append(dict(side=side, yaw=angle, x=float(pos[0]), y=float(pos[1]), reachable=ok))
    return dict(starts=starts, probes=probes)


def pinch_angles():
    """Recreate a central Lift grasp at four angles, using the real contact judge."""
    results = []
    with TabletopEnv(Lift) as env:
        for degrees in (0,15,30,45):
            env.reset(seed=11,finish_on_success=False,limit_ticks=False)
            xyz = np.array([.479, .235, .015])
            adr = env.task.free
            half_angle = np.deg2rad(degrees)/2
            env.data.qpos[adr:adr+7] = [*xyz,np.cos(half_angle),0,0,np.sin(half_angle)]
            vel = env.task.joints["red"].dofadr.item()
            env.data.qvel[vel:vel+6] = 0
            mujoco.mj_forward(env.model,env.data)
            ex = ArmExecutor(env,env.arm.observe(0,env.task.prompt),lambda *args: None)
            ex.home()
            calls = []
            for z, name in ((.075,"move_by"),(.017,"move_by"),(.017,"close_gripper"),(.075,"move_by")):
                args = {}
                if name == "move_by":
                    delta = 1000*(env.layout.local("right",xyz[:2],z)-env.arm.local_tcp())
                    args = dict(dx_mm=delta[0],dy_mm=delta[1],dz_mm=delta[2],grasp_yaw_deg=degrees)
                err = ex.execute(name,args)
                for _ in range(4):
                    if err != "SETTLE_TIMEOUT":
                        break
                    err = ex.execute("observe",{})
                calls.append([name,err])
                if err:
                    break
            results.append(dict(degrees=degrees,ticks=ex.obs.frame_no,calls=calls,**ex.info))
    return results


def main():
    out = Path("output/check_workspace")
    out.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB",(900,290),"white")
    draw = ImageDraw.Draw(canvas)
    for column, task in enumerate((Lift,Stack,Sort)):
        result = inspect(task)
        (out / (task.__name__.lower()+".json")).write_text(json.dumps(result,indent=2), encoding="utf8")
        cell_x = .300 if task is Sort else .479
        def pixel(x, y):
            return (column*300+150+(x-cell_x)*1320, 141-(y-.25)*1320)
        draw.rectangle([pixel(cell_x-(.078 if task is Sort else .083),.332),
                        pixel(cell_x+(.078 if task is Sort else .083),.168)],outline="black",width=2)
        draw.text((column*300+18,6),task.__name__+" | 16 seeded layouts",fill="black")
        for start in result["starts"]:
            for color,(x,y,degrees) in start["blocks"].items():
                box = square_corners([x,y],np.deg2rad(degrees))
                draw.polygon([pixel(*v) for v in box],outline="#b52328" if color=="red" else "#107448")
        draw.text((column*300+18,263),"black: tape inner edges",fill="black")
        print(task.__name__, sum(p["reachable"] for p in result["probes"]), "/",len(result["probes"]),flush=True)
    canvas.save(out/"workspace.png")
    angles = pinch_angles()
    (out/"angle_fragments.json").write_text(json.dumps(angles,indent=2),encoding="utf8")
    print("Physical Lift grasps:",[(a["degrees"],a["success"]) for a in angles],flush=True)


if __name__ == "__main__":
    main()
