"""Map rays from the overhead MuJoCo pinhole into the raw course front camera."""
from functools import lru_cache

import cv2
import numpy as np


FRONT_K = np.array([[570.21740069, 0, 327.45975405],
                    [0, 570.1797441, 260.83642155], [0, 0, 1]])
FRONT_D = np.array([-.735413911, .949258417, .000189059234, -.00200351391, -.864150312])


LENS_RADIUS = .70


def radial(r):
    k1, k2, _, _, k3 = FRONT_D
    return r * (1 + k1*r**2 + k2*r**4 + k3*r**6)


def radial_derivative(r):
    k1, k2, _, _, k3 = FRONT_D
    return 1 + 3*k1*r**2 + 5*k2*r**4 + 7*k3*r**6


def radial_distance(r):
    inner = np.minimum(r, LENS_RADIUS)
    return radial(inner) + (r-inner)*radial_derivative(LENS_RADIUS)


@lru_cache(maxsize=2)
def remap_grid(fovy):
    w, h = 640, 480
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    xd = (u - FRONT_K[0, 2]) / FRONT_K[0, 0]
    yd = (v - FRONT_K[1, 2]) / FRONT_K[1, 1]
    _, _, p1, p2, _ = FRONT_D
    # Beyond the checkerboard field the measured polynomial folds back on itself.
    # Extend its central branch tangentially instead of duplicating objects at the border.
    x, y = xd.copy(), yd.copy()
    for _ in range(7):
        r2 = x*x + y*y
        xu = xd - 2*p1*x*y - p2*(r2+2*x*x)
        yu = yd - p1*(r2+2*y*y) - 2*p2*x*y
        distance = np.hypot(xu, yu)
        r = np.sqrt(r2)
        inner = np.minimum(r, LENS_RADIUS)
        error = radial_distance(r) - distance
        r -= error/radial_derivative(inner)
        scale = r / np.maximum(distance, 1e-12)
        x, y = xu*scale, yu*scale
    rays = np.stack((x,y), axis=-1)
    focal = h / (2 * np.tan(np.radians(fovy) / 2))
    coords = (rays * focal + [(w-1)/2, (h-1)/2]).astype(np.float32)
    return cv2.convertMaps(coords, None, cv2.CV_16SC2)


def raw_front(rgb, fovy):
    maps = remap_grid(float(fovy))
    return cv2.remap(rgb, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
