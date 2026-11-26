import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from panner import calculate_gains
from effects import LowPassEffect, ReverbEffect, DopplerDelay

SR = 48000


def load_mono_48k(path):
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    mono = data.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        mono = resample_poly(mono, SR // g, sr // g)
    peak = np.max(np.abs(mono)) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / peak
    return np.ascontiguousarray(mono, dtype=np.float32)


class AudioEngine:
    def __init__(self):
        self.stream = None
        self.lock = threading.RLock()
        self.output_samples = 0
        self.source = np.zeros(1, dtype=np.float32)
        self.source_test = np.zeros(1, dtype=np.float32)
        self.frames = []
        self.frame_gains = []
        self.frame_rows = []
        self.mode = 'speaker'
        self.mapping = {'FL': 1, 'FR': 2, 'RL': 3, 'RR': 4}
        self.speaker_db = {'FL': 0, 'FR': 0, 'RL': 0, 'RR': 0}
        self.master_db = 0
        self.speaker_volume = {k: 10 ** (v / 20) for k, v in self.speaker_db.items()}
        self.master_volume = 1.0
        self.target_speaker_volume = dict(self.speaker_volume)
        self.target_master_volume = 1.0
        self.smooth_distance = 1.0
        self.settings = {}
        self.blocksize = 1024
        self.latency = 'high'
        self.device = None
        self.mode_run = 'stop'
        self.clipped = False
        self.underflow = False
        self.lowpass = [LowPassEffect(SR) for _ in range(4)]
        self.reverb = [ReverbEffect(SR) for _ in range(4)]
        self.doppler = [DopplerDelay(SR) for _ in range(4)]
        self.test_channel = None
        self.itd_ring = np.zeros((2, 64), dtype=np.float32)
        self.itd_write = 0

    def set_frames(self, frames):
        with self.lock:
            self.frames = frames

    def set_source(self, path):
        mono = load_mono_48k(path)
        with self.lock:
            self.source = mono

    def set_test_source(self, path):
        mono = load_mono_48k(path)
        with self.lock:
            self.source_test = mono

    def precompute_gains(self):
        with self.lock:
            settings = dict(self.settings)
            frames = list(self.frames)
        gains = []
        for row in frames:
            g, g_dist = calculate_gains(
                row['azimuth_deg'], row['elevation_deg'], row['distance_m'],
                float(settings.get('r0', 1.0)),
                float(settings.get('distance_exponent', 1.0)),
                float(settings.get('elevation_strength', 1.0)),
                str(settings.get('panning_law', '-3')))
            gains.append({name: value * g_dist for name, value in g.items()})
        with self.lock:
            self.frame_gains = gains
            self.frame_rows = frames

    def update_settings(self, values):
        with self.lock:
            old_settings = dict(self.settings)
            self.settings = dict(values)
            self.mode = values.get('mode', self.mode)
            self.mapping = dict(values.get('mapping', self.mapping))
            speaker_db = dict(values.get('speaker_db', self.speaker_db))
            master_db = float(values.get('master_db', self.master_db))
            self.speaker_db = speaker_db
            self.master_db = master_db
            self.target_speaker_volume = {
                k: 10 ** (float(v) / 20) for k, v in speaker_db.items()
            }
            self.target_master_volume = 10 ** (master_db / 20)
            need_recompute = any(values.get(k) != old_settings.get(k) for k in
                                 ('r0', 'distance_exponent', 'elevation_strength', 'panning_law'))
            self.validate_mapping()
        if need_recompute and self.frames:
            self.precompute_gains()

    def reset(self):
        with self.lock:
            self.output_samples = 0
            self.test_channel = None
            self.clipped = False
            self.underflow = False
            self.speaker_volume = dict(self.target_speaker_volume)
            self.master_volume = self.target_master_volume
            self.smooth_distance = float(self.settings.get('r0', 1.0))
            self.itd_ring.fill(0)
            self.itd_write = 0
            for effects in (self.lowpass, self.reverb, self.doppler):
                for effect in effects:
                    effect.reset()

    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.stream = None
        self.mode_run = 'stop'
        self.test_channel = None

    def _callback(self, outdata, _frames, _time_info, status):
        if status:
            if getattr(status, 'output_underflow', False) or getattr(status, 'underflow', False):
                self.underflow = True
        outdata.fill(0)
        if self.mode_run == 'test':
            self._test_callback(outdata)
        elif self.mode_run == 'trajectory':
            self._trajectory_callback(outdata, _frames)

    def _test_callback(self, outdata):
        n = len(outdata)
        start = self.output_samples
        with self.lock:
            source = self.source_test
            mapping = dict(self.mapping)
            speaker_db = dict(self.speaker_db)
            master_db = self.master_db
            mode = self.mode
        if source.size == 0:
            self.output_samples += n
            return

        if mode == 'earphone':
            segment = max(1, int(SR * 1.25))
            for i in range(n):
                absolute = start + i
                seg_index = (absolute // segment) % 2
                seg_pos = absolute % segment
                self.test_channel = seg_index + 1
                value = float(source[seg_pos]) if seg_pos < len(source) else 0.0
                if seg_index == 0:
                    outdata[i, 0] = value * (10 ** (master_db / 20))
                elif outdata.shape[1] >= 2:
                    outdata[i, 1] = value * (10 ** (master_db / 20))
            self.output_samples += n
            return

        segment = max(1, int(SR * 1.25))
        names = ('FL', 'FR', 'RL', 'RR')
        for i in range(n):
            absolute = start + i
            seg_index = (absolute // segment) % 4
            seg_pos = absolute % segment
            self.test_channel = seg_index + 1
            if seg_pos < len(source):
                value = float(source[seg_pos])
            else:
                value = 0.0
            name = names[seg_index]
            ch = mapping[name] - 1
            if mode == 'earphone':
                ch_out = 0 if seg_index in (0, 2) else 1
            else:
                ch_out = ch
            if 0 <= ch_out < outdata.shape[1]:
                outdata[i, ch_out] += value * (10 ** (speaker_db[name] / 20)) * (10 ** (master_db / 20))
        self.output_samples += n

    def _trajectory_callback(self, outdata, _frames):
        n = len(outdata)
        with self.lock:
            source = self.source
            frame_rows = list(self.frame_rows)
            settings = dict(self.settings)
            mapping = dict(self.mapping)
            mode = self.mode
            target_speaker = dict(self.target_speaker_volume)
            target_master = float(self.target_master_volume)
            renderer_method = str(settings.get('renderer_method', 'binaural'))

        if not source.size or not frame_rows:
            self.output_samples += n
            return

        r0 = float(settings.get('r0', 1.0))
        high_strength = float(settings.get('highpass_strength', 0.0))
        reverb_amount = float(settings.get('reverb_amount', 0.0))
        doppler_strength = float(settings.get('doppler_strength', 0.0)) if settings.get('doppler_enabled', False) else 0.0

        result = np.zeros((n, 4), dtype=np.float32)
        smooth_distance = float(getattr(self, 'smooth_distance', r0))
        names = ('FL', 'FR', 'RL', 'RR')
        frame_gains = list(self.frame_gains)

        for i in range(n):
            absolute = self.output_samples + i
            if absolute >= len(source):
                break

            frame_now = min(len(frame_rows) - 1, int(np.floor((absolute / SR) * 30)))
            frame_next = min(len(frame_rows) - 1, frame_now + 1)
            local = (absolute / SR * 30.0) - np.floor(absolute / SR * 30.0)
            gain_now = frame_gains[frame_now] if frame_gains else {name: 0.0 for name in names}
            gain_next = frame_gains[frame_next] if frame_gains else gain_now

            row = frame_rows[frame_now]
            distance = float(row['distance_m'])
            smooth_distance = smooth_distance * 0.9995 + distance * 0.0005

            base = float(source[absolute])
            for j, name in enumerate(names):
                alpha = 0.9995
                self.speaker_volume[name] = (self.speaker_volume[name] * alpha
                                             + target_speaker[name] * (1.0 - alpha))
                gain = gain_now[name] + (gain_next[name] - gain_now[name]) * local
                result[i, j] = base * gain * self.speaker_volume[name]

        self.smooth_distance = smooth_distance

        cutoff = 16000.0 * (r0 / max(smooth_distance, r0))
        reverb_mix = reverb_amount * min(1.0, smooth_distance / max(r0, 1.0))
        min_delay = r0 / 343.0
        doppler_delay = max(0.0, (smooth_distance / 343.0) - min_delay)

        for j in range(4):
            result[:, j] = self.lowpass[j].process(result[:, j], cutoff, high_strength)
            result[:, j] = self.reverb[j].process(result[:, j], reverb_mix)
            if doppler_strength > 0:
                result[:, j] = self.doppler[j].process(result[:, j], doppler_delay, doppler_strength)

        alpha_master = 0.9995
        self.master_volume = self.master_volume * alpha_master + target_master * (1.0 - alpha_master)
        result *= self.master_volume

        outdata.fill(0)
        if mode == 'earphone':
            if renderer_method == 'binaural':
                for i in range(n):
                    absolute = self.output_samples + i
                    frame_now = min(len(frame_rows) - 1, int(np.floor((absolute / SR) * 30)))
                    frame_next = min(len(frame_rows) - 1, frame_now + 1)
                    local = (absolute / SR * 30.0) - np.floor(absolute / SR * 30.0)
                    row_now = frame_rows[frame_now]
                    row_next = frame_rows[frame_next]
                    azimuth = float(row_now['azimuth_deg']) + (float(row_next['azimuth_deg']) - float(row_now['azimuth_deg'])) * local
                    elevation = float(row_now['elevation_deg']) + (float(row_next['elevation_deg']) - float(row_now['elevation_deg'])) * local
                    azimuth = max(-180.0, min(180.0, azimuth))
                    pan = (azimuth + 90.0) / 180.0
                    pan = max(0.0, min(1.0, pan))
                    left_gain = np.cos(pan * np.pi / 2.0)
                    right_gain = np.sin(pan * np.pi / 2.0)

                    mono = float(result[i].sum()) * 0.7071
                    sine = np.sin(np.radians(azimuth))
                    write = self.itd_write
                    self.itd_ring[0, write] = mono
                    self.itd_ring[1, write] = mono
                    self.itd_write = (write + 1) % 64
                    itd = min(int(round(abs(float(sine)) * 0.00065 * SR)), 63)
                    left_delay = itd if sine > 0 else 0
                    right_delay = itd if sine < 0 else 0
                    left_source = self.itd_ring[0, (write - left_delay) % 64]
                    right_source = self.itd_ring[1, (write - right_delay) % 64]

                    elevation_gain = 1.0 - 0.12 * abs(elevation) / 90.0
                    if elevation < 0:
                        elevation_gain *= 1.0 - (-elevation / 90.0) * 0.10
                    if outdata.shape[1] >= 1:
                        outdata[i, 0] = left_source * left_gain * elevation_gain
                    if outdata.shape[1] >= 2:
                        outdata[i, 1] = right_source * right_gain * elevation_gain
            else:
                for i in range(n):
                    left = result[i, 0] + result[i, 2]
                    right = result[i, 1] + result[i, 3]
                    if outdata.shape[1] >= 1:
                        outdata[i, 0] = left * 0.7071
                    if outdata.shape[1] >= 2:
                        outdata[i, 1] = right * 0.7071
        else:
            for j, name in enumerate(names):
                ch = mapping[name] - 1
                if 0 <= ch < outdata.shape[1]:
                    outdata[:, ch] += result[:, j]

        if np.max(np.abs(outdata)) > 1.0:
            self.clipped = True
            np.clip(outdata, -1.0, 1.0, out=outdata)
        self.output_samples += n

    def start_trajectory(self, device=None, blocksize=None, latency=None):
        self.reset()
        self.mode_run = 'trajectory'
        self.precompute_gains()
        channels = 2 if self.mode == 'earphone' else 4
        self.start(device, blocksize, latency, channels)

    def start_test(self, device=None, blocksize=None, latency=None):
        self.reset()
        self.mode_run = 'test'
        if self.mode == 'speaker':
            self.validate_mapping()
            channels = 4
        else:
            channels = 2
        self.start(device, blocksize, latency, channels)

    def validate_device(self, device, channels):
        try:
            info = sd.query_devices(device, 'output')
        except Exception as exc:
            raise ValueError(f"출력 장치를 확인할 수 없습니다: {exc}")
        max_channels = int(info['max_output_channels'])
        if max_channels < channels:
            requirement = '2채널 이어폰 또는 4채널 스피커 장치' if channels == 2 else '4채널 스피커 장치 또는 Aggregate Device'
            raise ValueError(
                f"선택한 장치 '{info['name']}'은(는) 최대 {max_channels}채널만 지원합니다. "
                f"이 모드에는 {requirement}가 필요합니다.")

    def start(self, device, blocksize, latency, channels):
        self.stop()
        self.output_samples = 0
        self.blocksize = int(blocksize)
        self.latency = latency
        if channels == 4:
            self.validate_mapping()
        self.validate_device(device, channels)
        self.stream = sd.OutputStream(device=device, samplerate=SR, channels=channels,
                                      dtype='float32', blocksize=self.blocksize,
                                      latency=self.latency, callback=self._callback)
        self.stream.start()

    def restart_output_mode(self, device=None, blocksize=None, latency=None):
        if not self.stream:
            return
        run_mode = self.mode_run
        position = self.output_samples
        channels = 2 if self.mode == 'earphone' else 4
        self.stop()
        self.mode_run = run_mode
        self.validate_device(device, channels)
        self.output_samples = position
        self.blocksize = int(blocksize)
        self.latency = latency
        self.stream = sd.OutputStream(device=device, samplerate=SR, channels=channels,
                                      dtype='float32', blocksize=self.blocksize,
                                      latency=self.latency, callback=self._callback)
        self.stream.start()

    def get_position(self):
        if not self.frames:
            return None
        sec = self.output_samples / SR
        frame = min(len(self.frames) - 1, int(np.floor(sec * 30)))
        return sec, frame, self.frames[frame]

    def devices(self):
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        result = []
        for i, d in enumerate(devices):
            if d['max_output_channels'] > 0:
                host = hostapis[d['hostapi']]['name']
                result.append((i, d['name'], d['max_output_channels'], host))
        return result

    def validate_mapping(self, mapping=None):
        values = mapping or self.mapping
        names = ('FL', 'FR', 'RL', 'RR')
        channels = [int(values[name]) for name in names]
        if any(ch < 1 or ch > 64 for ch in channels):
            raise ValueError('스피커 채널은 1부터 64 사이로 지정해야 합니다.')
        if len(set(channels)) != 4:
            raise ValueError('FL, FR, RL, RR은 서로 다른 출력 채널에 지정해야 합니다.')
