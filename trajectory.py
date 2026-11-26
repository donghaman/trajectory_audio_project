import math
from conventions import wrap_angle


class Trajectory:
    def __init__(self, frame_count=0, fps=30.0, width=0, height=0):
        self.frame_count = int(frame_count)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.origin_px = None
        self.front_angle_deg = 0.0
        self.keyframes = {}
        self.invisible_frames = set()

    def set_origin(self, x_px, y_px):
        self.origin_px = (float(x_px), float(y_px))

    def set_front_angle(self, angle_deg):
        self.front_angle_deg = wrap_angle(float(angle_deg))

    def set_keyframe(self, frame, x_px, y_px, visible=1):
        frame = int(frame)
        if frame < 0:
            raise ValueError('프레임은 0 이상이어야 합니다.')
        if visible:
            self.invisible_frames.discard(frame)
        else:
            self.invisible_frames.add(frame)
        self.keyframes[frame] = (float(x_px), float(y_px), int(bool(visible)))

    def modify_keyframe(self, frame, x_px=None, y_px=None, visible=None):
        if frame not in self.keyframes:
            raise KeyError('해당 프레임에 키프레임이 없습니다.')
        x, y, v = self.keyframes[frame]
        x = float(x if x_px is None else x_px)
        y = float(y if y_px is None else y_px)
        v = int(v if visible is None else visible)
        self.set_keyframe(frame, x, y, v)

    def delete_keyframe(self, frame):
        self.keyframes.pop(int(frame), None)

    def set_visible_range(self, start_frame, end_frame, visible):
        start = max(0, int(start_frame))
        end = min(max(start, int(end_frame)), max(0, self.frame_count - 1))
        for frame in range(start, end + 1):
            if frame in self.keyframes:
                x, y, _ = self.keyframes[frame]
                self.keyframes[frame] = (x, y, int(bool(visible)))
            if visible:
                self.invisible_frames.discard(frame)
            else:
                self.invisible_frames.add(frame)

    def interpolate_position(self, frame):
        frame = max(0, min(int(frame), max(0, self.frame_count - 1)))
        if frame in self.invisible_frames:
            return None

        key_values = {f: value for f, value in self.keyframes.items() if value[2] == 1}
        if frame in key_values:
            x, y, _ = key_values[frame]
            return x, y
        if not key_values:
            return None

        before = [f for f in key_values if f <= frame]
        after = [f for f in key_values if f >= frame]
        if before and after:
            f1 = max(before)
            f2 = min(after)
            if f1 == f2:
                x, y, _ = key_values[f1]
                return x, y
            x1, y1, _ = key_values[f1]
            x2, y2, _ = key_values[f2]
            p = (frame - f1) / (f2 - f1)
            return x1 + (x2 - x1) * p, y1 + (y2 - y1) * p

        if not before:
            f = min(key_values)
            x, y, _ = key_values[f]
            return x, y
        f = max(key_values)
        x, y, _ = key_values[f]
        return x, y

    def calculate_relative_position(self, frame):
        position = self.interpolate_position(frame)
        if position is None:
            return None
        if self.origin_px is None:
            raise RuntimeError('기준 위치가 설정되지 않았습니다.')

        x, y = position
        ox, oy = self.origin_px
        dx = x - ox
        dy = oy - y

        azimuth = wrap_angle(self.front_angle_deg + math.degrees(math.atan2(dx, max(dy, 1e-9))))
        pixel_distance = math.hypot(dx, dy)
        normalized_distance = 0.0
        if self.width > 0 and self.height > 0:
            normalized_distance = pixel_distance / math.hypot(self.width, self.height)
        return {
            'frame': frame,
            'time_s': frame / self.fps if self.fps > 0 else 0.0,
            'x_px': x,
            'y_px': y,
            'azimuth_deg': azimuth,
            'distance_px': pixel_distance,
            'distance': normalized_distance,
            'visible': 1,
        }

    def build_trajectory(self):
        rows=[]
        for frame in range(self.frame_count):
            if frame in self.invisible_frames:
                rows.append({
                    'frame': frame,
                    'time_s': frame / self.fps if self.fps > 0 else 0.0,
                    'x_px': None,
                    'y_px': None,
                    'azimuth_deg': None,
                    'distance': None,
                    'distance_px': None,
                    'visible': 0,
                })
                continue
            relative=self.calculate_relative_position(frame)
            if relative is None:
                rows.append({
                    'frame': frame,
                    'time_s': frame / self.fps if self.fps > 0 else 0.0,
                    'x_px': None,
                    'y_px': None,
                    'azimuth_deg': None,
                    'distance': None,
                    'distance_px': None,
                    'visible': 0,
                })
            else:
                rows.append(relative)
        return rows
