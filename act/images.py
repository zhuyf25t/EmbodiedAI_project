import numpy as np
from PIL import Image


def resize_rgb(rgb, image_hw):
    height, width = image_hw
    return np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR))
