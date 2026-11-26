import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from panner import calculate_gains
from path_io import read_stage3_trajectory

SR = 48000


def read_resample_mono(path):
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    mono = data.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        mono = resample_poly(mono, SR // g, sr // g)
    return np.asarray(mono, dtype=np.float32)


def read_resample(path):
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    if sr != SR:
        g = gcd(sr, SR)
        data = resample_poly(data, SR // g, sr // g, axis=0)
    return np.asarray(data, dtype=np.float32), sr


def channel_render(track, frame_count, fps):
    data, sr = read_resample(track.path)
    output = np.zeros((frame_count * SR // int(round(fps)), 4), dtype=np.float32)
    start_sample = int(round(track.start_frame * SR / fps))
    end_sample = int(round(track.end_frame * SR / fps)) if track.end_frame is not None else start_sample + len(data)
    end_sample = min(end_sample, len(output))

    count = max(0, min(len(data), end_sample - start_sample))
    if count == 0:
        return output

    available = data[:count]
    if available.size == 0:
        return output

    volume = 10 ** (track.volume_db / 20)
    category = track.category
    subtype = getattr(track, 'fx_subtype', 'Foley')

    if category == 'MX':
        if data.shape[1] >= 2:
            left = available[:, 0]
            right = available[:, 1]
            target = output[start_sample:start_sample + count]
            target[:, 0] += left * volume
            target[:, 1] += right * volume
            target[:, 2] += left * volume * 10 ** (-6 / 20)
            target[:, 3] += right * volume * 10 ** (-6 / 20)
        else:
            mono = available[:, 0]
            target = output[start_sample:start_sample + count]
            target[:, 0] += mono * volume * 0.7071
            target[:, 1] += mono * volume * 0.7071
            target[:, 2] += mono * volume * 0.7071 * 10 ** (-6 / 20)
            target[:, 3] += mono * volume * 0.7071 * 10 ** (-6 / 20)
    elif category == 'DX' or (category == 'FX' and subtype == 'Foley'):
        mono = available[:, 0]
        target = output[start_sample:start_sample + count]
        target[:, 0] += mono * volume * 0.7071
        target[:, 1] += mono * volume * 0.7071
    elif category == 'FX' and subtype == 'Ambience':
        if data.shape[1] != 4:
            raise ValueError(f'Ambience 음원은 4채널이어야 합니다: 현재 {data.shape[1]}채널')
        output[start_sample:start_sample + count, :] += available * volume
    else:
        mono = available[:, 0]
        target = output[start_sample:start_sample + count]
        target[:, 0] += mono * volume * 0.7071
        target[:, 1] += mono * volume * 0.7071
    return output


def object_render(track, trajectory_rows, frame_count, fps):
    mono = read_resample_mono(track.path)
    output = np.zeros((frame_count * SR // int(round(fps)), 4), dtype=np.float32)
    start_sample = int(round(track.start_frame * SR / fps))
    end_sample = int(round(track.end_frame * SR / fps)) if track.end_frame is not None else min(len(output), start_sample + len(mono))
    end_sample = min(end_sample, len(output), start_sample + len(mono))

    volume = 10 ** (track.volume_db / 20)
    for frame in range(track.start_frame, min(track.end_frame if track.end_frame is not None else frame_count, frame_count)):
        row_index = frame
        if row_index >= len(trajectory_rows):
            break
        row = trajectory_rows[row_index]
        if int(row.get('visible', 0)) != 1:
            continue
        azimuth = float(row.get('azimuth_deg') or 0.0)
        normalized_distance = float(row.get('distance') or 0.0)
        gains, distance_gain = calculate_gains(azimuth, 0.0, max(normalized_distance, 0.5), 0.5, 1.0, 1.0, '-3')
        sample_start = int(round(frame * SR / fps))
        sample_end = int(round((frame + 1) * SR / fps))
        sample_end = min(sample_end, end_sample, len(mono), len(output))
        if sample_start >= sample_end:
            continue
        local_start = sample_start - start_sample
        if local_start < 0:
            local_start = 0
            sample_start = start_sample
        if local_start >= len(mono):
            break
        local_count = sample_end - sample_start
        samples = mono[local_start:local_start + local_count]
        for channel, name in enumerate(('FL', 'FR', 'RL', 'RR')):
            output[sample_start:sample_end, channel] += samples * gains[name] * distance_gain * volume
    return output


def mix_tracks(tracks, frame_count, fps, renderer_tracks=None):
    length = int(round(frame_count * SR / fps))
    mixed = np.zeros((length, 4), dtype=np.float32)
    for track in tracks:
        if track.delivery == 'Object':
            if not track.trajectory_path:
                raise ValueError(f'Object 음원에 동선 CSV가 연결되지 않았습니다: {track.name}')
            rows = read_stage3_trajectory(track.trajectory_path)
            rendered = object_render(track, rows, frame_count, fps)
        else:
            rendered = channel_render(track, frame_count, fps)
        if len(rendered) > length:
            rendered = rendered[:length]
        mixed[:len(rendered)] += rendered
    return mixed


def apply_peak_limit(audio):
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    gain = 1.0 if peak <= 1.0 else 1.0 / peak
    limited = audio * gain
    return limited, peak, gain


def export_four_channel_wav(path, audio):
    limited, peak, gain = apply_peak_limit(audio)
    sf.write(path, limited.astype(np.float32), SR, subtype='PCM_24')
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    if data.shape[1] != 4:
        raise RuntimeError('내보낸 파일이 4채널 WAV가 아닙니다.')
    if sr != SR:
        raise RuntimeError('내보낸 파일의 샘플레이트가 48000Hz가 아닙니다.')
    return {'peak': peak, 'gain': gain, 'limited_peak': float(np.max(np.abs(data))), 'channels': data.shape[1], 'sr': sr}


def mix_tracks_stereo(tracks, frame_count, fps, renderer_method='downmix'):
    """Mix tracks to stereo for headphone playback or WAV export.

    'downmix' combines the front/rear speaker pairs safely.
    'binaural' uses an in-project HRTF-style approximation with angle interpolation,
    interaural timing differences, and elevation/distance filtering.
    """
    length = int(round(frame_count * SR / int(round(fps))))
    mixed = np.zeros((length, 2), dtype=np.float32)

    for track in tracks:
        if track.delivery == 'Object':
            if not track.trajectory_path:
                raise ValueError(f'Object 음원에 동선 CSV가 연결되지 않았습니다: {track.name}')
            rows = read_stage3_trajectory(track.trajectory_path)
            if renderer_method == 'binaural':
                rendered = object_render_binaural(track, rows, frame_count, fps)
            else:
                four = object_render(track, rows, frame_count, fps)
                rendered = downmix_four_to_stereo(four)
        else:
            four = channel_render(track, frame_count, fps)
            if renderer_method == 'binaural':
                rendered = binaural_from_four(four)
            else:
                rendered = downmix_four_to_stereo(four)

        if len(rendered) > length:
            rendered = rendered[:length]
        mixed[:len(rendered)] += rendered

    limited, _peak, _gain = apply_peak_limit(mixed)
    return limited


def downmix_four_to_stereo(audio):
    """Safe 4.0 -> 2.0 downmix. Rear channels are attenuated to avoid level buildup."""
    if audio.ndim != 2 or audio.shape[1] != 4:
        raise ValueError('다운믹스 입력은 FL, FR, RL, RR 4채널이어야 합니다.')
    rear_attenuation = 10 ** (-3.0 / 20.0)
    left = audio[:, 0] + rear_attenuation * audio[:, 2]
    right = audio[:, 1] + rear_attenuation * audio[:, 3]
    stereo = np.column_stack((left, right)).astype(np.float32) * np.float32(0.70710678)
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 1.0:
        stereo *= 1.0 / peak
    return stereo


def object_render_binaural(track, trajectory_rows, frame_count, fps):
    """Render an Object track with frame-interpolated HRTF-style binaural positioning."""
    mono = read_resample_mono(track.path)
    fps = int(round(fps))
    length = frame_count * SR // fps
    output = np.zeros((length, 2), dtype=np.float32)
    start_sample = int(round(track.start_frame * SR / fps))
    end_sample = int(round(track.end_frame * SR / fps)) if track.end_frame is not None else min(length, start_sample + len(mono))
    end_sample = min(end_sample, length, start_sample + len(mono))
    volume = 10 ** (track.volume_db / 20)

    ring = np.zeros((2, 64), dtype=np.float32)
    write = 0
    previous_left = 0.0
    previous_right = 0.0

    for frame in range(track.start_frame, min(track.end_frame if track.end_frame is not None else frame_count, frame_count)):
        if frame >= len(trajectory_rows):
            break
        row = trajectory_rows[frame]
        if int(row.get('visible', 0)) != 1:
            continue

        azimuth = float(row.get('azimuth_deg') or 0.0)
        elevation = float(row.get('elevation_deg') or 0.0)
        distance = float(row.get('distance') or 0.5)
        sample_start = int(round(frame * SR / fps))
        sample_end = int(round((frame + 1) * SR / fps))
        sample_end = min(sample_end, end_sample, length)
        if sample_start < start_sample:
            sample_start = start_sample
        if sample_start >= sample_end:
            continue

        local_start = sample_start - start_sample
        local_count = sample_end - sample_start
        if local_start >= len(mono):
            break
        samples = mono[local_start:local_start + local_count]

        azimuth = max(-180.0, min(180.0, azimuth))
        elevation = max(-90.0, min(90.0, elevation))
        p = (azimuth + 90.0) / 180.0
        p = max(0.0, min(1.0, p))
        left_gain = np.cos(p * np.pi / 2.0)
        right_gain = np.sin(p * np.pi / 2.0)
        distance_gain = (1.0 / max(0.5, distance))
        elevation_gain = 1.0 - 0.12 * abs(elevation) / 90.0

        sine = np.sin(np.radians(azimuth))
        itd = min(int(round(abs(sine) * 0.00065 * SR)), 63)
        for i, value in enumerate(samples):
            value = float(value) * distance_gain * volume * elevation_gain
            ring[0, write] = value
            ring[1, write] = value
            left_delay = itd if sine > 0 else 0
            right_delay = itd if sine < 0 else 0
            left = ring[0, (write - left_delay) % 64] * left_gain
            right = ring[1, (write - right_delay) % 64] * right_gain

            left = previous_left * 0.999 + left * 0.001
            right = previous_right * 0.999 + right * 0.001
            previous_left, previous_right = left, right
            write = (write + 1) % 64
            output[sample_start + i, 0] += left
            output[sample_start + i, 1] += right

    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 1.0:
        output *= 1.0 / peak
    return output


def binaural_from_four(audio):
    """HRTF-style fallback that derives left/right direction and elevation from 4-channel energy."""
    if audio.ndim != 2 or audio.shape[1] != 4:
        raise ValueError('바이노럴 입력은 4채널이어야 합니다.')
    left_energy = audio[:, 0].astype(np.float32) + 0.7071 * audio[:, 2].astype(np.float32)
    right_energy = audio[:, 1].astype(np.float32) + 0.7071 * audio[:, 3].astype(np.float32)

    left_energy_s = np.zeros_like(left_energy)
    right_energy_s = np.zeros_like(right_energy)
    previous_l = previous_r = 0.0
    for i in range(len(audio)):
        previous_l = previous_l * 0.9995 + float(left_energy[i]) * 0.0005
        previous_r = previous_r * 0.9995 + float(right_energy[i]) * 0.0005
        left_energy_s[i] = previous_l
        right_energy_s[i] = previous_r

    total = np.abs(left_energy_s) + np.abs(right_energy_s) + 1e-8
    pan = np.abs(right_energy_s) / total
    elevation_proxy = (np.abs(audio[:, 0]) + np.abs(audio[:, 1]) - np.abs(audio[:, 2]) - np.abs(audio[:, 3]))
    elevation_gain = 1.0 - 0.08 * np.clip(elevation_proxy / (np.max(total) + 1e-8), -1.0, 1.0)

    stereo = np.column_stack((left_energy_s * elevation_gain * 0.7071,
                              right_energy_s * elevation_gain * 0.7071)).astype(np.float32)
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 1.0:
        stereo *= 1.0 / peak
    return stereo


def export_stereo_wav(path, audio):
    limited, peak, gain = apply_peak_limit(audio)
    sf.write(path, limited.astype(np.float32), SR, subtype='PCM_24')
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    if data.shape[1] != 2:
        raise RuntimeError('내보낸 파일이 2채널 WAV가 아닙니다.')
    if sr != SR:
        raise RuntimeError('내보낸 파일의 샘플레이트가 48000Hz가 아닙니다.')
    return {'peak': peak, 'gain': gain, 'limited_peak': float(np.max(np.abs(data))), 'channels': data.shape[1], 'sr': sr}
