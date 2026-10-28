# Metal quality from tap vibrations (feature pipeline)

Companion to the SonicSense-style contact-mic rig. Extracts a ~24-dim feature vector
from one 16 kHz / 12-bit tap recording and trains leakage-safe baselines.

## Files
- `features.py` — load + validity check + preprocess + time/decay/frequency features
- `batch_extract.py` — folder/CSV -> features.csv
- `train_baseline.py` — group-aware (by object_id) train/test + LR/SVM/RF baselines

## Quick start
```
# 1) Extract features from a folder of tap recordings
python batch_extract.py --input path/to/taps --meta meta.csv --out features.csv

# 2) Train baselines (grouped by object, no tap-level leakage)
python train_baseline.py --csv features.csv --label quality --group object_id --avg
```

## Notes
- ADC-zeros / stuck-ADC recordings are rejected by `check_valid` (see features.py).
- For purity, use `--avg` (average taps per object) — single taps are too noisy.
- Split is by `object_id`, never by tap, so reported accuracy is honest.
- Keep feature extraction byte-identical between train and inference.
