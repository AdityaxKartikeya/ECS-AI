"""Capture one tap: from ESP32 over serial, or from a WAV folder."""
import glob
import os

import numpy as np


def _read_serial(cfg):
    import serial
    ser = serial.Serial(cfg["serial_port"], cfg["serial_baud"], timeout=5)
    n = int(cfg["sample_rate"] * cfg["tap_duration_ms"] / 1000)
    buf = bytearray()
    deadline_samples = n
    while len(buf) < deadline_samples * 2:
        chunk = ser.read(4096)
        if not chunk:
            break
        buf.extend(chunk)
    ser.close()
    if len(buf) < deadline_samples * 2:
        return None
    return np.frombuffer(bytes(buf[:deadline_samples * 2]), dtype="<i2")


def _read_wav(path):
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data[:, 0]
    return data, int(sr)


def record_one(cfg, wav_path=None):
    """Return (int16 array, sample_rate) or None on failure."""
    mode = cfg.get("input_mode", "serial")
    if mode == "serial":
        sig = _read_serial(cfg)
        if sig is None:
            return None
        return sig, int(cfg["sample_rate"])
    # wav mode
    if wav_path is None:
        files = sorted(glob.glob(os.path.join(cfg.get("wav_folder", "./taps"), "*.wav")))
        if not files:
            return None
        wav_path = files[0]
    try:
        data, sr = _read_wav(wav_path)
    except Exception:
        return None
    # convert to int16 if float
    if data.dtype.kind == "f":
        peak = float(np.max(np.abs(data))) or 1.0
        data = (data / peak * 3000).astype(np.int16)
    return data.astype(np.int16), sr
