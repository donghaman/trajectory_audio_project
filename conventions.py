import math


def wrap_angle(angle_deg: float) -> float:
    """Wrap to (-180, 180]."""
    wrapped = (angle_deg + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def angular_delta(from_deg: float, to_deg: float) -> float:
    """Signed shortest angular difference to - from, in degrees."""
    return wrap_angle(to_deg - from_deg)


def azimuth_to_xy(azimuth_deg: float, elevation_deg: float, distance_m: float):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    radial = distance_m * math.cos(el)
    return radial * math.sin(az), radial * math.cos(az)


def vertical_fov(horizontal_fov_deg: float, width: int, height: int) -> float:
    """Return vertical FOV for a pin-hole camera from aspect ratio."""
    if width <= 0 or height <= 0:
        raise ValueError('영상 너비와 높이는 양수여야 합니다.')
    h = math.radians(max(0.1, min(179.9, float(horizontal_fov_deg))))
    vfov = 2.0 * math.atan(math.tan(h / 2.0) * height / width)
    return math.degrees(vfov)


def screen_to_direction(click_x: float, click_y: float, width: int, height: int,
                       horizontal_fov_deg: float):
    """Convert a pixel click to azimuth/elevation using pin-hole geometry."""
    if width <= 0 or height <= 0:
        raise ValueError('영상 너비와 높이는 양수여야 합니다.')
    if not (0 <= click_x <= width and 0 <= click_y <= height):
        raise ValueError('클릭 좌표가 영상 범위 밖입니다.')

    nx = 2.0 * (float(click_x) / float(width)) - 1.0
    ny = 1.0 - 2.0 * (float(click_y) / float(height))
    hfov = math.radians(max(0.1, min(179.9, float(horizontal_fov_deg))))
    vfov = math.radians(vertical_fov(horizontal_fov_deg, width, height))

    azimuth = math.degrees(math.atan(nx * math.tan(hfov / 2.0)))
    elevation = math.degrees(math.atan(ny * math.tan(vfov / 2.0)))
    return azimuth, elevation, nx, ny
