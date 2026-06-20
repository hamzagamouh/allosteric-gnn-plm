"""Cross-validation training with a Random Forest classifier.

Usage examples:
    python -m ligand_classification.train_rf --model maccs
    python -m ligand_classification.train_rf --model esm+dpocket --folds 0 1 2 3 4
    python -m ligand_classification.train_rf --model dpocket --n-estimators 100 --output-dir results/

Available models: maccs | dpocket | esm | esm+dpocket | esm+dpocket+maccs
"""
import argparse
import json
import os
import numpy as np
from collections import defaultdict

from ligand_classification.config import FEATURES_DIR, N_FOLDS
from ligand_classification.data.io import (
    load_fold_arrays, clean_dpocket_nans, compose_features, get_feature_names,
)
from ligand_classification.models.rf import train_rf
from ligand_classification.models.metrics import balance_subsample, build_val_arrays

AVAILABLE_MODELS = ["maccs", "dpocket", "esm", "esm+dpocket", "esm+dpocket+maccs"]


def run(model, folds, features_dir, n_estimators, output_dir):
    fold_train_mccs, fold_val_mccs = [], []
    fold_importances = []
    feature_names = None

    for fold in folds:
        print(f"\n--- fold {fold} [{model}] ---")
        arrays = clean_dpocket_nans(load_fold_arrays(fold, features_dir))

        if feature_names is None:
            feature_names = get_feature_names(model, arrays)

        train_allo = compose_features(arrays, model, "train", "allo")
        train_non_allo = compose_features(arrays, model, "train", "non_allo")
        val_allo = compose_features(arrays, model, "val", "allo")
        val_non_allo = compose_features(arrays, model, "val", "non_allo")

        train_feats, y_train = balance_subsample(train_allo, train_non_allo)
        val_feats, y_val = build_val_arrays(val_allo, val_non_allo)

        tm, vm, importances = train_rf(train_feats, y_train, val_feats, y_val, n_estimators=n_estimators)
        print(f"  train_mcc={tm:.4f}  val_mcc={vm:.4f}")

        fold_train_mccs.append(tm)
        fold_val_mccs.append(vm)
        fold_importances.append(importances)

    mean_imp = np.mean(fold_importances, axis=0)
    top10_idx = np.argsort(mean_imp)[::-1][:10]
    top10 = [
        {"rank": int(i + 1), "feature": feature_names[idx], "importance": float(mean_imp[idx])}
        for i, idx in enumerate(top10_idx)
    ]

    print(f"\n===== RESULTS [{model}] =====")
    print(f"train_mcc: {np.mean(fold_train_mccs):.3f} ± {np.std(fold_train_mccs):.3f}")
    print(f"val_mcc:   {np.mean(fold_val_mccs):.3f} ± {np.std(fold_val_mccs):.3f}")
    print("\nTop 10 features:")
    for entry in top10:
        print(f"  {entry['rank']:2d}. {entry['feature']:40s} {entry['importance']:.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        slug = model.replace("+", "_")

        results = {
            "model": model,
            "folds": folds,
            "n_estimators": n_estimators,
            "train_mcc_per_fold": fold_train_mccs,
            "val_mcc_per_fold": fold_val_mccs,
            "train_mcc_mean": float(np.mean(fold_train_mccs)),
            "train_mcc_std": float(np.std(fold_train_mccs)),
            "val_mcc_mean": float(np.mean(fold_val_mccs)),
            "val_mcc_std": float(np.std(fold_val_mccs)),
        }
        json.dump(results, open(f"{output_dir}/{slug}_cv_results.json", "w"), indent=2)
        json.dump(top10, open(f"{output_dir}/{slug}_top10_features.json", "w"), indent=2)
        print(f"\nSaved results to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", required=True, choices=AVAILABLE_MODELS,
        help="Feature set to train on.",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(N_FOLDS)))
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--features-dir", default=FEATURES_DIR)
    parser.add_argument("--output-dir", default=None, help="Directory to save results JSON files.")
    args = parser.parse_args()

    run(args.model, args.folds, args.features_dir, args.n_estimators, args.output_dir)


if __name__ == "__main__":
    main()
