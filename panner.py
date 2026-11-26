import math
from conventions import wrap_angle, angular_delta

SPEAKER_AZIMUTH = {"FL": -30.0, "FR": 30.0, "RL": -135.0, "RR": 135.0}
SPEAKER_PAIRS = (("FL", "FR", -30.0, 30.0),
                 ("FR", "RR", 30.0, 135.0),
                 ("RR", "RL", 135.0, -135.0),
                 ("RL", "FL", -135.0, -30.0))


def _pair_values(p: float, law: str):
    p = min(1.0, max(0.0, p))
    if law == "-6":
        g1, g2 = 1.0 - p, p
    else:
        g1, g2 = math.cos(p * math.pi / 2.0), math.sin(p * math.pi / 2.0)
        if law == "-4.5":
            lin1, lin2 = 1.0 - p, p
            g1 = math.sqrt(max(g1 * lin1, 0.0))
            g2 = math.sqrt(max(g2 * lin2, 0.0))
    return g1, g2


def horizontal_gains(azimuth_deg: float, panning_law: str = "-3"):
    az = wrap_angle(float(azimuth_deg))
    gains = {name: 0.0 for name in SPEAKER_AZIMUTH}

    for name1, name2, angle1, angle2 in SPEAKER_PAIRS:
        span = angular_delta(angle1, angle2)
        offset = angular_delta(angle1, az)
        if span >= 0.0:
            inside = 0.0 <= offset <= span
        else:
            inside = span <= offset <= 0.0
        if inside:
            p = offset / span if span else 0.0
            g1, g2 = _pair_values(p, panning_law)
            gains[name1], gains[name2] = g1, g2

    return gains


def calculate_gains(azimuth_deg: float, elevation_deg: float, distance_m: float,
                    r0_m: float = 1.0, distance_exponent: float = 1.0,
                    elevation_strength: float = 1.0,
                    panning_law: str = "-3"):
    horizontal = horizontal_gains(azimuth_deg, panning_law)
    el = max(-90.0, min(90.0, float(elevation_deg)))
    w = float(elevation_strength) * math.sin(math.radians(el))
    w = max(-1.5, min(1.5, w))

    gains = {}
    for name, g in horizontal.items():
        g_sq = (1.0 - w) * (g * g) + w / 4.0
        gains[name] = math.sqrt(max(g_sq, 0.0))

    r0 = max(0.5, float(r0_m))
    distance = max(float(distance_m), r0)
    g_dist = (r0 / distance) ** max(0.0, float(distance_exponent))
    return gains, g_dist
