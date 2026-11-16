"""Orbit height from an observed trunk segment, in the same local/world frame."""
import math


def orbit_target_z(base_z, height, fraction=0.5, minimum=1.0, maximum=4.0):
    values=(base_z,height,fraction,minimum,maximum)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite trunk estimate or orbit limit')
    if height <= 0 or not 0 < fraction <= 1 or minimum <= 0 or maximum < minimum:
        raise ValueError('Invalid trunk estimate or orbit limit')
    return base_z+max(minimum,min(maximum,height*fraction))
