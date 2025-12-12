import math


def lr_cosine_schedule(t: int, lr_max: float, lr_min: float, T_w: int, T_c: int):
    if t < T_w:
        return t / T_w * lr_max
    if t > T_c:
        return lr_min
    return lr_min + 0.5 * (1 + math.cos((t-T_w)/(T_c-T_w) * math.pi)) * (lr_max - lr_min)