"""Baseline training: group-aware split by object_id, LR / SVM / RandomForest."""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from features import FEATURE_ORDER

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", required=True, help="target label column")
    ap.add_argument("--group", default="object_id", help="group column for split (no leakage)")
    ap.add_argument("--avg", action="store_true", help="average taps per group before training (recommended for purity)")
    ap.add_argument("--test-size", type=float, default=0.3)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    feat_cols = [c for c in FEATURE_ORDER if c in df.columns]
    X = df[feat_cols].values
    y = df[args.label].values
    groups = df[args.group].values if args.group in df.columns else np.arange(len(df))

    # optionally average per group
    if args.avg and args.group in df.columns:
        gdf = df.copy()
        gdf["_g"] = groups
        agg = gdf.groupby(args.group, as_index=False).agg({**{c: "mean" for c in feat_cols}, args.label: "first"})
        X = agg[feat_cols].values
        y = agg[args.label].values
        groups = agg[args.group].values
        print(f"Averaged to {len(X)} group-level samples")

    # group-aware split
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    models = {
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "SVM-RBF": SVC(kernel="rbf", class_weight="balanced", probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    }
    for name, m in models.items():
        if name == "RandomForest":
            m.fit(Xtr, ytr); pred = m.predict(Xte)
        else:
            m.fit(Xtr_s, ytr); pred = m.predict(Xte_s)
        print(f"\n===== {name} =====")
        print(classification_report(yte, pred, zero_division=0))
        print("F1 (macro):", round(f1_score(yte, pred, average="macro", zero_division=0), 4))
        print("Confusion matrix:\n", confusion_matrix(yte, pred))

if __name__ == "__main__":
    main()
