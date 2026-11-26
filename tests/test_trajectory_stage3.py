import csv
import pytest

from trajectory import Trajectory
from path_io import save_stage3_trajectory, read_stage3_trajectory


def test_origin_and_right_side_azimuth():
    trajectory = Trajectory(frame_count=11, fps=30, width=1000, height=500)
    trajectory.set_origin(500, 250)
    trajectory.set_front_angle(0)
    trajectory.set_keyframe(0, 500, 250)
    trajectory.set_keyframe(10, 750, 250)
    right = trajectory.calculate_relative_position(10)
    assert right['azimuth_deg'] == pytest.approx(90.0)


def test_linear_interpolation():
    trajectory = Trajectory(frame_count=11, fps=30)
    trajectory.set_keyframe(0, 100, 100)
    trajectory.set_keyframe(10, 200, 300)
    assert trajectory.interpolate_position(5) == pytest.approx((150.0, 200.0))


def test_delete_and_modify_keyframe():
    trajectory = Trajectory(frame_count=5)
    trajectory.set_keyframe(2, 100, 100)
    trajectory.modify_keyframe(2, 120, 130, 1)
    assert trajectory.keyframes[2] == (120.0, 130.0, 1)
    trajectory.delete_keyframe(2)
    assert 2 not in trajectory.keyframes


def test_visible_range():
    trajectory = Trajectory(frame_count=10)
    trajectory.set_keyframe(0, 100, 100)
    trajectory.set_visible_range(3, 6, False)
    for frame in range(3, 7):
        assert frame in trajectory.invisible_frames
        assert trajectory.interpolate_position(frame) is None


def test_visible_keyframe_saved_as_visible_flag():
    trajectory = Trajectory(frame_count=3)
    trajectory.set_keyframe(1, 10, 20, False)
    assert trajectory.keyframes[1][2] == 0


def test_distance_unit_is_normalized_not_meters():
    trajectory = Trajectory(frame_count=2, width=1000, height=500)
    trajectory.set_origin(500, 250)
    trajectory.set_keyframe(1, 600, 250)
    row = trajectory.calculate_relative_position(1)
    assert row['distance_px'] == pytest.approx(100.0)
    assert row['distance'] == pytest.approx(100 / (1000**2 + 500**2)**0.5)


def test_stage3_csv_round_trip(tmp_path):
    rows = [{
        'frame': 0, 'time_s': 0.0, 'x_px': 100.0, 'y_px': 200.0,
        'azimuth_deg': 10.0, 'distance': 0.2, 'distance_px': 200.0, 'visible': 1,
    }]
    path = tmp_path / 'trajectory.csv'
    save_stage3_trajectory(str(path), rows)
    loaded = read_stage3_trajectory(str(path))
    assert loaded == rows


def test_legacy_csv_is_compatible(tmp_path):
    path = tmp_path / 'legacy.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['frame','time_s','azimuth_deg','elevation_deg','distance_m','source'])
        writer.writeheader()
        writer.writerow({'frame':0,'time_s':0,'azimuth_deg':0,'elevation_deg':0,'distance_m':2,'source':'given'})
    rows = read_stage3_trajectory(str(path))
    assert rows[0]['visible'] == 1
    assert rows[0]['azimuth_deg'] == 0.0
