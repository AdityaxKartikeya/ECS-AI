"""Train a baseline on a features CSV and export artifacts for the server."""
import argparse, json, os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from features import FEATURE_ORDER

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="demo/features.csv")
    ap.add_argument("--label", default="quality")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    feat_cols = [c for c in FEATURE_ORDER if c in df.columns]
    X = df[feat_cols].values
    y = df[args.label].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42)
    model.fit(Xs, y)

    classes = list(model.classes_)
    with open(os.path.join(args.out, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(args.out, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.out, "feature_columns.json"), "w") as f:
        json.dump(feat_cols, f)
    with open(os.path.join(args.out, "label_classes.json"), "w") as f:
        json.dump(classes, f)

    # quick sanity
    acc = model.score(Xs, y)
    print(f"Trained on {len(X)} samples, {len(feat_cols)} features, {len(classes)} classes")
    print("Classes:", classes)
    print("Train acc (in-sample, sanity only):", round(acc, 4))
    print("Artifacts saved to", os.path.abspath(args.out))

if __name__ == "__main__":
    main()
