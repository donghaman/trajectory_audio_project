import pytest

from conventions import screen_to_direction, vertical_fov


def test_screen_center_is_zero():
    az, el, nx, ny = screen_to_direction(960, 540, 1920, 1080, 90)
    assert az == pytest.approx(0.0, abs=1e-9)
    assert el == pytest.approx(0.0, abs=1e-9)
    assert nx == pytest.approx(0.0)
    assert ny == pytest.approx(0.0)


def test_right_and_left_direction_sign():
    right = screen_to_direction(1920, 540, 1920, 1080, 90)[0]
    left = screen_to_direction(0, 540, 1920, 1080, 90)[0]
    assert right > 0
    assert left < 0


def test_up_and_down_direction_sign():
    up = screen_to_direction(960, 0, 1920, 1080, 90)[1]
    down = screen_to_direction(960, 1080, 1920, 1080, 90)[1]
    assert up > 0
    assert down < 0


def test_fov_change_changes_result():
    narrow = screen_to_direction(1440, 540, 1920, 1080, 60)[0]
    wide = screen_to_direction(1440, 540, 1920, 1080, 120)[0]
    assert wide > narrow


def test_resolution_independent_normalization():
    az1 = screen_to_direction(75, 50, 100, 100, 90)[0]
    az2 = screen_to_direction(150, 100, 200, 200, 90)[0]
    assert az1 == pytest.approx(az2)


def test_vertical_fov_from_aspect_ratio():
    vfov = vertical_fov(90, 1920, 1080)
    assert vfov == pytest.approx(58.7, abs=0.2)


def test_click_outside_raises():
    with pytest.raises(ValueError, match='범위 밖'):
        screen_to_direction(-1, 540, 1920, 1080, 90)


def test_project_video_can_be_opened_and_read():
    import glob
    import os

    project_root = os.path.dirname(os.path.dirname(__file__))
    paths = glob.glob(os.path.join(project_root, 'Sample', '**', '*.mp4'), recursive=True)
    if not paths:
        pytest.skip('샘플 MP4가 없습니다.')

    from video_engine import VideoEngine
    engine = VideoEngine(paths[0])
    frame = engine.read_frame(0)
    assert frame.shape[0] > 0
    assert frame.shape[1] > 0
    assert frame.shape[2] == 3
    engine.close()
