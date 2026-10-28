"""Local pre-send validation (mirrors server-side check_valid)."""
import numpy as np


def check_valid(sig, sr):
    n = len(sig)
    if n < int(0.02 * sr):
        return False, "too_short"
    if np.count_nonzero(sig) / n < 0.05:
        return False, "mostly_zeros"
    if len(np.unique(sig)) < 5:
        return False, "stuck_adc"
    if np.max(np.abs(sig.astype(np.float64))) >= 4090:
        return False, "clipped"
    return True, "ok"
