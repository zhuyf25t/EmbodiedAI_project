from pathlib import Path

import numpy as np

from .cubes import TwoCubes


class Stack(TwoCubes):
    prompt = "用右臂将红块叠在绿块正中，松爪后让两块稳定。"
    model_file = Path(__file__).parent / "assets/so101/stack.xml"

    def arrangement(self, model, data, pairs):
        red, green = (self.cubes[c] for c in self.colors)
        delta = data.geom_xpos[red] - data.geom_xpos[green]
        offset = float(np.linalg.norm(delta[:2]))
        supported = (frozenset((red, green)) in pairs and frozenset((green, self.table)) in pairs
                     and abs(delta[2] - .030) <= .003)
        return supported and offset <= .008, dict(center_error_mm=offset * 1000, red_supported=bool(supported))
