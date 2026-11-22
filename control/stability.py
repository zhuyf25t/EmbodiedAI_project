"""RGB stability test reused from MuJoCo-VISTA-ACT control/stability.py."""
import cv2
import numpy as np


def compare_images(previous: np.ndarray, current: np.ndarray) -> bool:
    """Mean absolute RGB differences in 3x3 pixel blocks, including partial edges.

    Local windows overlap at every block, so motion cannot hide at tile boundaries.
    A high global similarity alone does not certify a stable image.
    """
    if (not isinstance(previous, np.ndarray) or not isinstance(current, np.ndarray)
            or previous.dtype != np.uint8 or current.dtype != np.uint8
            or previous.ndim != 3 or previous.shape != current.shape
            or previous.shape[2] != 3 or min(previous.shape[:2]) < 1):
        raise ValueError("Image comparison requires equal-sized nonempty uint8 RGB images")
    block = 3
    height, width = previous.shape[:2]
    ys, xs = np.arange(0, height, block), np.arange(0, width, block)
    # Subtract before pooling: opposite changes inside one block cannot cancel.
    difference = np.abs(current.astype(np.float32) - previous.astype(np.float32))
    sums = np.add.reduceat(np.add.reduceat(difference, ys, axis=0), xs, axis=1)
    area = np.minimum(block, height - ys)[:, None] * np.minimum(block, width - xs)[None, :]
    changed = (sums / area[..., None]).max(axis=2) > 5.0
    count = int(changed.sum())
    similarity = 1.0 - count / changed.size
    side = max(1, 24 // block)
    rows, columns = min(side, changed.shape[0]), min(side, changed.shape[1])
    integral = np.pad(changed.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    local_counts = (integral[rows:, columns:] - integral[:-rows, columns:]
                    - integral[rows:, :-columns] + integral[:-rows, :-columns])
    local_ratio = float(local_counts.max() / (rows * columns))
    if similarity >= 0.998 and local_ratio <= 0.10:
        return True
    # At macro focus a subpixel edge can flicker across a 24 px block. Accept it
    # only if both full-resolution frames agree within one pixel everywhere.
    if difference.mean() > .15:
        return False
    for source, target in ((previous, current), (current, previous)):
        above = cv2.subtract(target, cv2.dilate(source, None))
        below = cv2.subtract(cv2.erode(source, None), target)
        if max(above.max(), below.max()) > 5:
            return False
    return True
