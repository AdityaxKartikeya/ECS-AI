"""Batch feature extraction over a folder/CSV of tap recordings."""
import argparse, csv, os, sys
import numpy as np
import pandas as pd
from features import extract_from_file, FEATURE_ORDER

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of recordings OR a CSV with a 'filepath' column")
    ap.add_argument("--meta", default=None, help="optional metadata CSV (filepath,object_id,material,quality,trial_id,...)")
    ap.add_argument("--out", default="features.csv", help="output CSV path")
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    # build list of files
    files = []
    if os.path.isdir(args.input):
        for root, _, fs in os.walk(args.input):
            for f in fs:
                if f.lower().endswith((".wav", ".npy", ".bin", ".raw", ".pcm")):
                    files.append(os.path.join(root, f))
    else:
        # treat as CSV
        df = pd.read_csv(args.input)
        if "filepath" not in df.columns:
            print("CSV must contain a 'filepath' column"); sys.exit(1)
        files = df["filepath"].tolist()

    # load metadata if provided
    meta = {}
    if args.meta:
        mdf = pd.read_csv(args.meta)
        for _, r in mdf.iterrows():
            meta[os.path.basename(str(r["filepath"]))] = r.to_dict()

    rows = []
    n_ok = 0; n_bad = 0
    for fp in files:
        try:
            vec, status = extract_from_file(fp, args.sr)
        except Exception as e:
            status = f"error:{e}"; vec = None
        if vec is None:
            n_bad += 1
            print(f"SKIP {fp} -> {status}")
            continue
        n_ok += 1
        row = {"filepath": fp}
        for i, name in enumerate(FEATURE_ORDER):
            row[name] = float(vec[i])
        if os.path.basename(fp) in meta:
            for k, v in meta[os.path.basename(fp)].items():
                if k != "filepath":
                    row[k] = v
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"\nDone: {n_ok} ok, {n_bad} skipped -> {args.out}")
    print(f"Feature columns: {len(FEATURE_ORDER)}")

if __name__ == "__main__":
    main()
