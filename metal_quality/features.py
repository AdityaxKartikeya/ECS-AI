"""Feature extraction for one metal tap recording (16 kHz, 12-bit, one tap)."""
import numpy as np
from scipy.signal import hilbert, find_peaks
from scipy.stats import kurtosis, skew

FS_DEFAULT = 16000
NYQUIST = FS_DEFAULT / 2.0
# Keep a headroom below Nyquist for anti-alias safety
FMAX_USABLE = 7000.0

BANDS = [(100, 500), (500, 1500), (1500, 4000), (4000, 7000)]


def load_signal(path, sr_expected=FS_DEFAULT):
    """Load a tap recording. Supports .wav, .npy, and raw int16 .bin/.raw/.pcm."""
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        sig = np.load(path)
        return np.asarray(sig).ravel().astype(np.float64), int(sr_expected)
    if ext == ".wav":
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        data = np.asarray(data)
        if data.ndim > 1:
            data = data[:, 0]
        return data.astype(np.float64), int(sr)
    # raw int16 little-endian
    raw = np.fromfile(path, dtype="<i2")
    return raw.astype(np.float64), int(sr_expected)


def check_valid(sig, sr):
    """Return (ok, reason). Guards against the ADC-zeros / stuck-ADC problem."""
    n = len(sig)
    if n < int(0.02 * sr):
        return False, "too_short"
    nz = np.count_nonzero(sig)
    if nz / n < 0.05:
        return False, "mostly_zeros"
    # stuck at a single value (ADC not converting)
    if len(np.unique(sig)) < 5:
        return False, "stuck_adc"
    # clipping at 12-bit full scale (allow a little headroom)
    if np.max(np.abs(sig)) >= 4090:
        return False, "clipped"
    return True, "ok"


def preprocess(sig, sr):
    """DC removal + light high-pass to kill slow drift; returns float64."""
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - np.mean(sig)
    # gentle high-pass at ~20 Hz via simple differencing-ish: use scipy if available
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 20.0 / (sr / 2.0), btype="highpass")
        sig = filtfilt(b, a, sig)
    except Exception:
        pass
    # normalize to unit peak (keeps features scale-invariant to tap force-ish)
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak
    return sig


def time_features(sig):
    rms = float(np.sqrt(np.mean(sig ** 2)))
    peak = float(np.max(np.abs(sig)))
    pp = float(np.max(sig) - np.min(sig))
    crest = float(peak / rms) if rms > 0 else 0.0
    energy = float(np.sum(sig ** 2))
    zcr = float(np.mean(np.diff(np.sign(sig)) != 0))
    k = float(kurtosis(sig, fisher=True, bias=False))
    s = float(skew(sig, bias=False))
    return {
        "rms": rms, "peak": peak, "peak2peak": pp, "crest": crest,
        "energy": energy, "zcr": zcr, "kurtosis": k, "skew": s,
        "std": float(np.std(sig)),
    }


def decay_features(sig, sr):
    env = np.abs(hilbert(sig))
    # smooth envelope a little
    w = max(1, int(0.002 * sr))
    kernel = np.ones(w) / w
    env_s = np.convolve(env, kernel, mode="same")
    # locate peak of envelope
    i_peak = int(np.argmax(env_s))
    tail = env_s[i_peak:]
    # time vector for tail
    t = np.arange(len(tail)) / sr
    # 10% time
    thr10 = 0.10 * env_s[i_peak]
    idx10 = np.where(tail <= thr10)[0]
    t10 = float(t[idx10[0]]) if len(idx10) else float(t[-1])
    # exponential fit on log envelope (clip to avoid log(0))
    logenv = np.log(np.clip(tail, 1e-6, None))
    # fit only until envelope hits 5% of peak (or first 100 ms)
    mask = tail >= (0.05 * env_s[i_peak])
    if mask.sum() < 5:
        mask = np.ones(len(tail), dtype=bool)
        mask[int(0.1 * sr):] = False
        if mask.sum() < 5:
            mask = np.ones(len(tail), dtype=bool)
    try:
        coef = np.polyfit(t[mask], logenv[mask], 1)
        slope = float(coef[0])
        tau = float(-1.0 / slope) if slope < 0 else float("inf")
    except Exception:
        tau = float("nan")
    # Q factor of dominant resonance: f_peak / bandwidth (half-power)
    # computed in frequency_features caller; placeholder here
    T60 = float(6.908 * tau) if np.isfinite(tau) else float("nan")
    # time for the envelope to fall to 5% of its peak
    thr5 = 0.05 * env_s[i_peak]
    idx5 = np.where(tail <= thr5)[0]
    t5 = float(t[idx5[0]]) if len(idx5) else float(t[-1])
    return {
        "decay_tau": tau, "T60": T60, "time_to_10pct": t10,
        "decay_time_5pct": t5,
        "env_peak": float(env_s[i_peak]),
    }


def frequency_features(sig, sr):
    n = len(sig)
    # windowed FFT
    win = np.hanning(n)
    X = np.abs(np.fft.rfft(sig * win))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    # usable band
    usable = freqs <= FMAX_USABLE
    f_u = freqs[usable]
    X_u = X[usable]
    total = float(np.sum(X_u ** 2) + 1e-12)

    # dominant peaks
    peaks, props = find_peaks(X_u, height=np.max(X_u) * 0.1)
    peak_freqs = f_u[peaks]
    peak_amps = X_u[peaks]
    order = np.argsort(peak_amps)[::-1]
    top = [(float(peak_freqs[i]), float(peak_amps[i])) for i in order[:5]]
    dom_freq = top[0][0] if top else 0.0

    # Q of dominant: half-power bandwidth
    Q = 0.0
    if top:
        f0, a0 = top[0]
        half = a0 / np.sqrt(2.0)
        left = f0
        right = f0
        # search nearest indices
        idx0 = int(np.argmin(np.abs(f_u - f0)))
        j = idx0
        while j > 0 and X_u[j] >= half:
            j -= 1
        left = f_u[j]
        j = idx0
        while j < len(f_u) - 1 and X_u[j] >= half:
            j += 1
        right = f_u[j]
        bw = max(right - left, 1e-3)
        Q = float(f0 / bw)

    # spectral centroid
    centroid = float(np.sum(f_u * X_u) / (np.sum(X_u) + 1e-12))
    # bandwidth (std around centroid)
    bandwidth = float(np.sqrt(np.sum(((f_u - centroid) ** 2) * X_u) / (np.sum(X_u) + 1e-12)))
    # rolloff 85%
    cum = np.cumsum(X_u ** 2)
    thr = 0.85 * cum[-1]
    idx_r = np.searchsorted(cum, thr)
    rolloff = float(f_u[min(idx_r, len(f_u) - 1)])
    # band energies (as ratios to total)
    band_ratios = []
    for lo, hi in BANDS:
        m = (f_u >= lo) & (f_u < hi)
        e = float(np.sum(X_u[m] ** 2))
        band_ratios.append(e / total)
    # spectral flatness
    geo = np.exp(np.mean(np.log(X_u + 1e-12)))
    ari = np.mean(X_u)
    flatness = float(geo / (ari + 1e-12))
    # harmonic ratios
    har = []
    if len(top) >= 2:
        for i in range(1, min(3, len(top))):
            har.append(top[i][0] / (top[0][0] + 1e-9))
    while len(har) < 2:
        har.append(0.0)

    out = {
        "dom_freq": dom_freq,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "rolloff85": rolloff,
        "spectral_flatness": flatness,
        "Q_factor": Q,
        "har_ratio_1": har[0],
        "har_ratio_2": har[1],
    }
    for i, (lo, hi) in enumerate(BANDS):
        out[f"band_ratio_{i}"] = band_ratios[i]
    # store top peak freqs
    for i in range(5):
        out[f"peak_freq_{i}"] = top[i][0] if i < len(top) else 0.0
        out[f"peak_amp_{i}"] = top[i][1] if i < len(top) else 0.0
    return out


FEATURE_ORDER = [
    "rms", "peak", "peak2peak", "crest", "energy", "zcr", "kurtosis", "skew",
    "decay_tau", "T60", "time_to_10pct", "env_peak",
    "dom_freq", "spectral_centroid", "spectral_bandwidth", "rolloff85",
    "spectral_flatness", "Q_factor", "har_ratio_1", "har_ratio_2",
    "band_ratio_0", "band_ratio_1", "band_ratio_2", "band_ratio_3",
    "peak_freq_0", "peak_freq_1", "peak_freq_2",
]


# --- peer contract -----------------------------------------------------------
# The friend's server publishes one fixed 20-key vector; everything downstream
# (his model, our dashboard) reads exactly these names in exactly this order.
PEER_FEATURE_ORDER = [
    "rms", "peak", "p2p", "crest", "energy", "std", "zcr",
    "decay_time_5pct", "decay_tau", "dom_freq", "centroid", "bandwidth",
    "rolloff85", "flatness", "peak2_freq", "peak3_freq",
    "band_frac_0", "band_frac_1", "band_frac_2", "band_frac_3",
]

# peer key -> our internal FEATURE_ORDER key (identity entries left implicit)
PEER_RENAME = {
    "p2p": "peak2peak",
    "centroid": "spectral_centroid",
    "bandwidth": "spectral_bandwidth",
    "flatness": "spectral_flatness",
    "peak2_freq": "peak_freq_1",
    "peak3_freq": "peak_freq_2",
    "band_frac_0": "band_ratio_0",
    "band_frac_1": "band_ratio_1",
    "band_frac_2": "band_ratio_2",
    "band_frac_3": "band_ratio_3",
}


def peer_features(feats):
    """Map our internal feature dict onto the peer's 20-key contract.

    Keys come out in PEER_FEATURE_ORDER and every value is a finite float.
    """
    out = {}
    for pk in PEER_FEATURE_ORDER:
        v = feats.get(PEER_RENAME.get(pk, pk), 0.0)
        v = float(v)
        out[pk] = v if np.isfinite(v) else 0.0
    return out


def extract_full(sig, sr=FS_DEFAULT):
    """Return (vec, feats_dict, reason). vec is None when the tap is invalid."""
    ok, reason = check_valid(sig, sr)
    if not ok:
        return None, None, reason
    sig = preprocess(sig, sr)
    feats = {}
    feats.update(time_features(sig))
    feats.update(decay_features(sig, sr))
    feats.update(frequency_features(sig, sr))
    # assemble ordered vector
    vec = np.array([feats.get(k, 0.0) for k in FEATURE_ORDER], dtype=np.float64)
    # replace non-finite
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, feats, "ok"


def extract(sig, sr=FS_DEFAULT):
    """Return (vec, reason) — model vector only, unchanged for training/inference."""
    vec, _feats, reason = extract_full(sig, sr)
    return vec, reason


def extract_from_file(path, sr_expected=FS_DEFAULT):
    sig, sr = load_signal(path, sr_expected)
    if sr != sr_expected:
        # resample if needed
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, sr_expected)
        sig = resample_poly(sig, sr_expected // g, sr // g)
        sr = sr_expected
    return extract(sig, sr)
