import numpy as np
import soundfile as sf
import pytest

from audio_tracks import AudioTrack
from renderer import channel_render, apply_peak_limit, mix_tracks


def test_track_duplicate_and_unique_id():
    a = AudioTrack(1, 'a.wav')
    b = AudioTrack(2, 'a.wav')
    assert a.id != b.id
    assert a.path == b.path


def test_dx_channel_center_rule(tmp_path):
    path = tmp_path / 'dx.wav'
    sf.write(path, np.ones(4800, dtype=np.float32), 48000)
    track = AudioTrack(1, str(path), category='DX', start_frame=0, end_frame=1)
    out = channel_render(track, frame_count=1, fps=30)
    assert out[:, 0].max() > 0
    assert out[:, 1].max() > 0
    assert out[:, 2].max() == 0
    assert out[:, 3].max() == 0


def test_mx_mono_is_mapped_front(tmp_path):
    path = tmp_path / 'mx.wav'
    sf.write(path, np.ones(4800, dtype=np.float32), 48000)
    track = AudioTrack(1, str(path), category='MX', start_frame=0, end_frame=1)
    out = channel_render(track, frame_count=1, fps=30)
    assert out[:, 0].max() > 0
    assert out[:, 1].max() > 0


def test_ambience_requires_four_channels(tmp_path):
    path = tmp_path / 'amb.wav'
    sf.write(path, np.ones((4800, 2), dtype=np.float32), 48000)
    track = AudioTrack(1, str(path), category='FX', delivery='Channel', start_frame=0, end_frame=1, fx_subtype='Ambience')
    with pytest.raises(ValueError, match='4채널'):
        channel_render(track, frame_count=1, fps=30)


def test_peak_limit_is_applied():
    audio = np.ones((100, 4), dtype=np.float32) * 2
    limited, peak, gain = apply_peak_limit(audio)
    assert peak == pytest.approx(2.0)
    assert gain == pytest.approx(0.5)
    assert np.max(np.abs(limited)) == pytest.approx(1.0)


def test_mix_tracks_sums_and_is_four_channel(tmp_path):
    path = tmp_path / 'tone.wav'
    sf.write(path, np.ones(4800, dtype=np.float32), 48000)
    track = AudioTrack(1, str(path), category='DX', start_frame=0, end_frame=1)
    mixed = mix_tracks([track], frame_count=1, fps=30)
    assert mixed.shape[1] == 4


def test_export_four_channel_wav(tmp_path):
    from renderer import export_four_channel_wav
    path = tmp_path / 'out.wav'
    audio = np.ones((4800, 4), dtype=np.float32)
    result = export_four_channel_wav(str(path), audio)
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    assert data.shape[1] == 4
    assert sr == 48000
    assert result['gain'] == pytest.approx(1.0)


def test_audio_track_json_round_trip():
    from audio_tracks import AudioTrack
    track = AudioTrack(7, 'voice.wav', category='DX', delivery='Object', start_frame=12, end_frame=120, volume_db=-3.0, trajectory_path='voice.csv')
    restored = AudioTrack.from_dict(track.to_dict())
    assert restored.to_dict() == track.to_dict()


def test_downmix_is_stereo_and_peak_limited():
    from renderer import downmix_four_to_stereo
    audio = np.ones((100, 4), dtype=np.float32)
    stereo = downmix_four_to_stereo(audio)
    assert stereo.shape == (100, 2)
    assert np.max(np.abs(stereo)) <= 1.0


def test_export_stereo_wav(tmp_path):
    from renderer import export_stereo_wav
    path = tmp_path / 'out.wav'
    audio = np.ones((4800, 2), dtype=np.float32)
    result = export_stereo_wav(str(path), audio)
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    assert data.shape[1] == 2
    assert sr == 48000
    assert result['channels'] == 2


def test_object_binaural_azimuth_moves_left_right(tmp_path):
    from renderer import object_render_binaural
    import csv

    wav_path = tmp_path / 'tone.wav'
    sf.write(wav_path, np.ones(4800, dtype=np.float32), 48000)
    csv_path = tmp_path / 'trajectory.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['frame','time_s','x_px','y_px','azimuth_deg','distance','distance_px','visible','elevation_deg'])
        writer.writeheader()
        for frame, azimuth in ((0, -45.0), (1, 45.0)):
            writer.writerow({'frame': frame, 'time_s': frame / 30.0, 'x_px': 0, 'y_px': 0,
                             'azimuth_deg': azimuth, 'distance': 1.0, 'distance_px': 100,
                             'visible': 1, 'elevation_deg': 0.0})

    track = AudioTrack(1, str(wav_path), delivery='Object', start_frame=0,
                       end_frame=2, trajectory_path=str(csv_path))
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        import csv as csv_module
        rows = list(csv_module.DictReader(f))
    for row in rows:
        row['visible'] = int(row['visible'])
        row['azimuth_deg'] = float(row['azimuth_deg'])
        row['distance'] = float(row['distance'])
        row['elevation_deg'] = float(row['elevation_deg'])

    stereo = object_render_binaural(track, rows, frame_count=2, fps=30)
    assert stereo.shape == (3200, 2)
    assert np.max(stereo[:1600, 0]) > np.max(stereo[:1600, 1])
    assert np.max(stereo[1600:, 1]) > np.max(stereo[1600:, 0])
