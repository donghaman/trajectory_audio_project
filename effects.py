import numpy as np


class LowPassEffect:
    def __init__(self, sr=48000):
        self.sr = sr
        self.state = 0.0

    def reset(self):
        self.state = 0.0

    def process(self, samples, cutoff_hz, strength):
        if strength <= 0:
            return samples.copy()
        cutoff = max(100.0, min(float(cutoff_hz), self.sr * 0.45))
        alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff / self.sr)
        out = np.empty_like(samples)
        for i, x in enumerate(samples):
            self.state += alpha * (x - self.state)
            out[i] = (1.0 - strength) * x + strength * self.state
        return out


class ReverbEffect:
    def __init__(self, sr=48000):
        self.buffers = [np.zeros(int(sr * 0.12), dtype=np.float32) for _ in range(4)]
        self.delays = [29, 37, 47, 59]
        self.pos = 0

    def reset(self):
        for buf in self.buffers:
            buf.fill(0)
        self.pos = 0

    def process(self, samples, amount):
        if amount <= 0:
            return samples.copy()
        out = np.empty_like(samples)
        feedback = 0.35
        for i, x in enumerate(samples):
            wet = 0.0
            for bi, ms in enumerate(self.delays):
                buf = self.buffers[bi]
                delay = max(1, int(buf.size * ms / 120.0))
                idx = (self.pos - delay) % buf.size
                y = buf[idx]
                buf[self.pos] = x + feedback * y
                wet += y
            self.pos = (self.pos + 1) % len(self.buffers[0])
            out[i] = x + amount * 0.25 * wet
        return out


class DopplerDelay:
    def __init__(self, sr=48000):
        self.sr = sr
        self.buf = np.zeros(int(sr * 0.1), dtype=np.float32)
        self.write = 0

    def reset(self):
        self.buf.fill(0)
        self.write = 0

    def process(self, samples, delay_seconds, strength):
        if strength <= 0:
            return samples.copy()
        out = np.empty_like(samples)
        float_delay = max(0.0, min(float(delay_seconds), 0.09)) * self.sr
        for i, x in enumerate(samples):
            self.buf[self.write] = x
            read = self.write - float_delay
            low = int(np.floor(read))
            frac = read - low
            y1 = self.buf[low % len(self.buf)]
            y2 = self.buf[(low + 1) % len(self.buf)]
            y = y1 + frac * (y2 - y1)
            out[i] = (1.0 - strength) * x + strength * y
            self.write = (self.write + 1) % len(self.buf)
        return out
