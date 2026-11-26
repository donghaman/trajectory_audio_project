import csv
import numpy as np

REQUIRED_COLUMNS = ["frame", "time_s", "azimuth_deg", "elevation_deg", "distance_m", "source"]
STAGE3_COLUMNS = [
    "frame", "time_s", "x_px", "y_px", "azimuth_deg",
    "elevation_deg", "distance", "distance_px", "visible",
]


def read_trajectory(path: str):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV 헤더를 찾을 수 없습니다.")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError("CSV 열 이름이 예상과 다릅니다. 누락 열: " + ", ".join(missing)
                             + "\n실제 열: " + ", ".join(reader.fieldnames))

        frames = []
        for line, row in enumerate(reader, start=2):
            try:
                frames.append({
                    "frame": int(float(row["frame"])),
                    "time_s": float(row["time_s"]),
                    "azimuth_deg": float(row["azimuth_deg"]),
                    "elevation_deg": float(row["elevation_deg"]),
                    "distance_m": float(row["distance_m"]),
                    "source": row["source"],
                })
            except (TypeError, ValueError) as exc:
                raise ValueError(f"CSV {line}번째 줄의 숫자 형식이 올바르지 않습니다: {exc}")

    if not frames:
        raise ValueError("CSV에 데이터 행이 없습니다.")
    frames.sort(key=lambda item: item["frame"])
    for i, item in enumerate(frames):
        if i != item["frame"]:
            raise ValueError("frame은 0부터 시작해 1씩 증가해야 합니다.")
        if not (1e-3 <= item["distance_m"] <= 1e6):
            raise ValueError("distance_m은 양수여야 합니다.")
        if not -180.0 < item["azimuth_deg"] <= 180.0:
            raise ValueError("azimuth_deg은 (-180, 180] 범위여야 합니다.")

    return frames


def trajectory_arrays(frames):
    return (np.array([r["frame"] for r in frames], dtype=int),
            np.array([r["time_s"] for r in frames], dtype=float),
            np.array([r["azimuth_deg"] for r in frames], dtype=float),
            np.array([r["elevation_deg"] for r in frames], dtype=float),
            np.array([r["distance_m"] for r in frames], dtype=float))


def save_stage3_trajectory(path, rows):
    fieldnames = list(STAGE3_COLUMNS)
    if not any(row.get('elevation_deg') is not None for row in rows):
        fieldnames.remove('elevation_deg')
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {name: row.get(name, '') for name in fieldnames}
            if output['visible'] in (None, ''):
                output['visible'] = 0
            writer.writerow(output)


def read_stage3_trajectory(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError('CSV 헤더를 찾을 수 없습니다.')
        fields = set(reader.fieldnames)

        if {'frame', 'time_s', 'x_px', 'y_px', 'azimuth_deg', 'distance', 'visible'} <= fields:
            rows=[]
            for line, row in enumerate(reader, start=2):
                try:
                    frame=int(float(row['frame']))
                    visible=int(float(row['visible']))
                    rows.append({
                        'frame': frame,
                        'time_s': float(row['time_s']),
                        'x_px': None if visible == 0 or not row['x_px'] else float(row['x_px']),
                        'y_px': None if visible == 0 or not row['y_px'] else float(row['y_px']),
                        'azimuth_deg': None if visible == 0 or not row['azimuth_deg'] else float(row['azimuth_deg']),
                        'distance': None if visible == 0 or not row['distance'] else float(row['distance']),
                        'distance_px': None if visible == 0 or not row.get('distance_px', '') else float(row['distance_px']),
                        'visible': visible,
                    })
                    if 'elevation_deg' in fields:
                        rows[-1]['elevation_deg'] = 0.0 if visible == 0 or not row.get('elevation_deg', '') else float(row['elevation_deg'])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"CSV {line}번째 줄 형식 오류: {exc}")
            return rows

        if set(REQUIRED_COLUMNS) <= fields:
            legacy=read_trajectory(path)
            return [{
                'frame': row['frame'],
                'time_s': row['time_s'],
                'x_px': None,
                'y_px': None,
                'azimuth_deg': row['azimuth_deg'],
                'elevation_deg': row['elevation_deg'],
                'distance': row['distance_m'],
                'distance_px': None,
                'visible': 1,
            } for row in legacy]

        raise ValueError(
            '동선 CSV 형식을 알 수 없습니다.\n'
            '신규 열: frame,time_s,x_px,y_px,azimuth_deg,distance,visible\n'
            f'실제 열: {", ".join(reader.fieldnames)}')
