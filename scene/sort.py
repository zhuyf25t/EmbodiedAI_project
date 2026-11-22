from pathlib import Path

import numpy as np

from .cubes import TwoCubes


class Sort(TwoCubes):
    dual = True
    carry_height = .060
    prompt = "将公共初始区的两块分拣，右臂放红块入图纸右格，左臂放绿块入图纸左格，松爪后保持稳定。"
    model_file = Path(__file__).parent / "assets/so101/sort.xml"

    def arrangement(self, model, data, pairs):
        inside = {}
        for color, cube in self.cubes.items():
            zone = model.site(color + "_target").id
            half = np.abs(data.geom_xmat[cube].reshape(3, 3)[:2]) @ model.geom_size[cube]
            offset = np.abs(data.geom_xpos[cube, :2] - data.site_xpos[zone, :2])
            inside[color] = bool(np.all(offset + half <= model.site_size[zone, :2])
                                 and frozenset((cube, self.table)) in pairs)
        return all(inside.values()), dict(inside=inside)
